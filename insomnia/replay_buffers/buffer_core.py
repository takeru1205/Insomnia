from collections import namedtuple


class BaseBuffer:
    """The Base of Experience Replay.
    """
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

    def store_transition(self):
        """To store state transition
        """
        raise NotImplementedError

    def sample_buffer(self):
        """Sampling data
        """
        raise NotImplementedError

    def __len__(self):
        """Return data length
        """
        raise NotImplementedError
