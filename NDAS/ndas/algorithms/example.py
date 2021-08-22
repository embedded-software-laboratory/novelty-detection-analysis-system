from ndas.algorithms.basedetector import BaseDetector


class ExampleDetector(BaseDetector):

    def __init__(self, *args, **kwargs):
        super(ExampleDetector, self).__init__(*args, **kwargs)

    def detect(self, datasets, **kwargs) -> dict:
        result = {}

        for column in datasets.columns[1:]:
            novelty_data = {}

            i = 0
            for index, row in datasets[[datasets.columns[0], column]].iterrows():
                if i == 0:
                    novelty_data[row[datasets.columns[0]]] = 1
                    i = 1
                else:
                    novelty_data[row[datasets.columns[0]]] = 0
                    i = 0

            result[column] = novelty_data
        return result

