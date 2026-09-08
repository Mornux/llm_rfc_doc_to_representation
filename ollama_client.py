import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  #use the local ollama api
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))  #get the project folder
MODELS_PATH = os.path.join(PROJECT_PATH, "models")  #store local model data here
OLLAMA_HEALTH_URL = "http://127.0.0.1:11434/api/tags"


def find_ollama_executable(): #find the local ollama executable without using a cloud service
    #prefer an executable already available on the system path
    executable = shutil.which("ollama")  #ask windows where the executable is located
    if executable:
        return executable

    #check common windows installation folders
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    candidates = [
        os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe"),
        os.path.join(local_app_data, "Ollama", "ollama.exe"),
        os.path.join(program_files, "Ollama", "ollama.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    #return the expected user installation path for a useful error message
    if local_app_data:
        return candidates[0]
    return None


def ensure_ollama_server(startup_timeout=20): #start the local ollama server when it is not already running
    #reuse ollama when the health endpoint is already available
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=1):
            return
    except (urllib.error.URLError, TimeoutError):
        pass

    #find the executable before attempting to start a server
    executable = find_ollama_executable()
    if not executable:
        raise RuntimeError(
            "Ollama is not running and ollama.exe could not be found. "
            "Install Ollama or add it to PATH."
        )

    environment = os.environ.copy()
    environment["OLLAMA_MODELS"] = MODELS_PATH
    #hide the ollama console window on windows
    creation_flags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    #start the server without attaching it to this process input or output
    try:
        subprocess.Popen(
            [executable, "serve"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except OSError as error:
        raise RuntimeError("Failed to start local Ollama: " + str(error)) from error

    #repeat the health request until the server is ready
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=1):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)

    raise RuntimeError(
        "Local Ollama did not start within " + str(startup_timeout) + " seconds."
    )


def load_models(): #return every model ollama can call including cloud models
    #make sure the local api is available before requesting model tags
    ensure_ollama_server()
    #read and decode the model list returned by ollama
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Failed to list Ollama models: " + str(error)) from error

    #collect unique non-empty model names
    model_list = []
    for model_data in result.get("models", []):
        model_name = model_data.get("name")
        if not model_name:
            model_name = model_data.get("model")
        if isinstance(model_name, str):
            model_name = model_name.strip()
            if model_name and model_name not in model_list:
                model_list.append(model_name)

    #keep model ordering stable across program runs
    model_list.sort(key=str.lower)
    return model_list


def is_cloud_model(model_name): #return whether an ollama model is a cloud model
    #ollama marks remote models with the cloud suffix
    return str(model_name).lower().endswith(":cloud")


def run_model(model_name, user_text, json_output=False): #send text to a model registered with ollama
    #reject missing model names before sending a generation request
    available_models = load_models()
    if model_name not in available_models:
        available_text = "none"
        if available_models:
            available_text = ", ".join(available_models)
        raise RuntimeError(
            "Ollama model is not available: " + model_name
            + ". Available models: " + available_text
        )

    #build one non-streaming generation request
    request_data = {
        "model": model_name,
        "prompt": user_text,
        "stream": False,
    }
    if json_output: #request syntactically valid json from ollama
        request_data["format"] = "json"
    json_data = json.dumps(request_data).encode("utf-8")  #encode the request body

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json_data,
        headers={"Content-Type": "application/json"},
    )

    #send the request and return only the generated response text
    try:
        with urllib.request.urlopen(request, timeout=3000) as response:  #call ollama
            result = json.loads(response.read().decode("utf-8"))
        response_text = result.get("response", "")
        if response_text:
            return response_text

        #keep thinking content when a reasoning model returns no response text
        thinking_text = result.get("thinking", "")
        if thinking_text:
            return thinking_text
        return ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Failed to run model " + model_name + ": " + str(error)
        ) from error
