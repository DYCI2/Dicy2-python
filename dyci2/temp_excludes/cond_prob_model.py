from collections import Counter

from model import Model


class CondProb(Model):
    def __init__(self, sequence=[], labels=[], equiv=(lambda x, y: x == y), label_type=None, content_type=None):
        """Constructor for the class CondProb.
        The sequence must be a list of elements with the form (event,conditions)
        """
        self.counts = Counter()
        self.conditions_counts = Counter()
        super(CondProb, self).__init__(sequence, labels, equiv, label_type, content_type)

    def learn_event(self, state, label, equiv=None):
        if equiv is None:
            equiv = self.equiv

        try:
            assert len(state) == 2
        except AssertionError as exception:
            print("The event", state, "of the sequence is not of the form (event,conditions)", exception)
            return None
        else:
            (event, conditions) = state
            self.counts.update([(event, conditions)])
            self.conditions_counts.update(conditions)

    def get_counts(self, state):
        return self.counts[state]

    def get_conditions_counts(self, conditions):
        return self.conditions_counts[conditions]