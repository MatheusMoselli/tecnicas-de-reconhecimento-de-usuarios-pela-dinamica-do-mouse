from src.classifiers    import EnumClassifiers
from src.dataset_loaders import EnumDatasets
from src.preprocessors  import EnumPreprocessors
from src.splitters      import EnumSplitters
from src.orchestrator   import Orchestrator

ALL_DATASETS = [EnumDatasets.MINECRAFT, EnumDatasets.BALABIT, EnumDatasets.BOGAZICI]
ALL_CLASSIFIERS  = [EnumClassifiers.KNN, EnumClassifiers.MLP, EnumClassifiers.RANDOM_FOREST]
ALL_WINDOW_SIZES = [200, 150, 100, 50, 10]
ALL_SEEDS = [1, 2, 3, 4, 5]


for dataset in ALL_DATASETS:
    for seed in ALL_SEEDS:
        for window_size in ALL_WINDOW_SIZES:
            orchestrator = Orchestrator(
                dataset=dataset,
                splitter=EnumSplitters.HALF,
                classifiers=ALL_CLASSIFIERS,
                preprocessor_window_size=window_size,
                preprocessor=EnumPreprocessors.KHAN,
                seed_number=seed,
                is_debug=False,
            )
            
            orchestrator.run()