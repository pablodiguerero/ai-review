class StreamInterrupted(Exception):
    def __init__(self, error: Exception, consumed: bool):
        self.error = error
        self.consumed = consumed

        super().__init__(str(error))
