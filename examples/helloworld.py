from tinker import protocol


class MyProtocol(protocol.Protocol):
    def get_config(self):
        return {}

    def run(self):
        print("hello, world")
