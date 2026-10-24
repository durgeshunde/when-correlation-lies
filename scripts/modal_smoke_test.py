import modal

app = modal.App("collusion-smoke-test")

@app.function()
def hello():
    import platform
    return f"Hello from Modal! Running on {platform.system()} {platform.machine()}"

@app.local_entrypoint()
def main():
    res = hello.remote()
    print("Output:", res)
