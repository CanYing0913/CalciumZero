# Message Structure for inter-process communication
class Message:
    def __init__(self, idx: int, is_running: bool = None, is_finished: bool = None):
        super.__setattr__(self, 'dictionary', dict())
        self.dictionary['idx'] = idx
        self.dictionary['is_running'] = is_running
        self.dictionary['is_finished'] = is_finished

    def __getattr__(self, item):
        try:
            return self.dictionary[item]
        except KeyError:
            raise AttributeError(f"Message has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key == 'dictionary':
            super().__setattr__(key, value)
        else:
            self.dictionary[key] = value

    def __setitem__(self, key, value):
        if key == 'dictionary':
            super().__setattr__(key, value)
        else:
            self.dictionary[key] = value

    def __str__(self):
        return str(self.dictionary)


message = {
    'idx': 0, 'is_running': False, 'is_finished': False
}
