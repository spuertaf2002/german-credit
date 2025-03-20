from pandas import DataFrame
from pyconfparser import ConfigFactory
from pyconfparser.config import Config

from src.config.config_schema import ConfigSchema
from src.pipeline.extraction import Extraction
from src.pipeline.transformation import Transformation

if __name__ == "__main__":
    config: Config = ConfigFactory.get_conf(r"src/config/config.json", ConfigSchema)
    extraction: DataFrame = Extraction(config.extraction).run()
    transformation: DataFrame = Transformation(config.transformation).run(extraction)

    print(transformation.info())
