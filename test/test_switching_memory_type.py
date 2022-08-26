from dyci2.corpus_event import Dyci2CorpusEvent
from dyci2.generation_scheduler import Dyci2GenerationScheduler
from dyci2.label import ListLabel, ChordLabel
from dyci2.prospector import Dyci2Prospector, FactorOracleProspector
from merge.main.corpus import GenericCorpus

if __name__ == '__main__':
    corpus: GenericCorpus = GenericCorpus([Dyci2CorpusEvent(i + 100, i, label=ListLabel.parse(str(i))) for i in range(10)], label_types=[ListLabel])
    prospector: Dyci2Prospector = FactorOracleProspector(corpus=corpus, label_type=ListLabel)
    generation_scheduler: Dyci2GenerationScheduler = Dyci2GenerationScheduler(prospector=prospector)
    print(generation_scheduler.generator.prospector.label_type)
    print(len(generation_scheduler.generator.prospector.corpus))

    corpus: GenericCorpus = GenericCorpus([], label_types=[ChordLabel])
    generation_scheduler.read_memory(corpus, override=True)
    generation_scheduler.learn_event(Dyci2CorpusEvent(123, 0, ChordLabel.parse("c maj")))

    print(f"{1+1}")


