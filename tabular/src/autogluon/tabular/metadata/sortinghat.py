import logging

import pandas as pd
import sortinghat.pylib as pl

from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.features.infer_types import get_type_map_raw, get_type_group_map_special, get_type_map_special, \
    get_type_group_map
from autogluon.core.features.types import R_INT, R_FLOAT, R_OBJECT, R_CATEGORY, R_DATETIME, S_TEXT, R_BOOL, S_BOOL
from autogluon.tabular.metadata.metadata_engine import MetadataEngine

logger = logging.getLogger(__name__)


class SortingHatFeatureMetadataEngine(MetadataEngine):

    """
    SortingHat inference types:
        Numeric: 0
        Categorical: 1
        Datetime: 2
        Sentence: 3
        URL: 4
        Numbers: 5
        List: 6
        Not - Generalizable: 7
        Custom
        Object( or Context - Specific): 8
    AutoGluon feature types:
        R_INT = 'int'
        R_FLOAT = 'float'
        R_OBJECT = 'object'
        R_CATEGORY = 'category'
        R_DATETIME = 'datetime'
        R_BOOL = 'bool'
        S_BOOL = 'bool'
        S_BINNED = 'binned'
        S_DATETIME_AS_INT = 'datetime_as_int'
        S_DATETIME_AS_OBJECT = 'datetime_as_object'
        S_TEXT = 'text'
        S_TEXT_AS_CATEGORY = 'text_as_category'
        S_TEXT_SPECIAL = 'text_special'
        S_TEXT_NGRAM = 'text_ngram'
        S_IMAGE_PATH = 'image_path'
        S_STACK = 'stack'
    """

    def infer_feature_metadata(self, df: pd.DataFrame):
        data_featurized = pl.FeatureExtraction(pl.FeaturizeFile(df))
        data_featurized = data_featurized.fillna(0)
        y_RF = pl.Load_RF(data_featurized)
        return self.to_feature_metadata(df, y_RF)

    def to_feature_metadata(self, df: pd.DataFrame, y_RF):
        features = [col for col in df]
        type_map_sh = {features[i]: y_RF[i] for i in range(len(features))}
        type_map_raw_default = get_type_map_raw(df)
        type_map_special_default = get_type_map_special(df)
        type_map_raw = self._get_raw_type_map(features, type_map_sh, type_map_raw_default)
        type_map_special = self._get_special_type_map(features, type_map_sh, type_map_special_default)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=get_type_group_map(type_map_special))

    def _get_raw_type_map(self, features, type_map_sh, type_map_raw_default):
        type_map_raw = {}
        for f in features:
            sh_inferred_type = type_map_sh[f]
            # category
            if sh_inferred_type == 1:
                type_map_raw[f] = R_BOOL if type_map_raw_default[f] == R_BOOL else R_CATEGORY
            # datetime
            elif sh_inferred_type == 2:
                type_map_raw[f] = R_DATETIME
            # object
            elif sh_inferred_type in {3, 4, 5, 6, 7, 8}:
                type_map_raw[f] = R_OBJECT
            else:
                # numeric -> fallback to AG's type inference to determine whether it's R_INT or R_FLOAT
                # others -> fall back to AG's inferred types
                type_map_raw[f] = type_map_raw_default[f]
        return type_map_raw

    def _get_special_type_map(self, features, type_map_sh, type_map_special_default):
        type_map_special = {}
        for f in features:
            sh_inferred_type = type_map_sh[f]
            # datetime
            if sh_inferred_type == 2:
                if f in type_map_special_default:
                    type_map_special[f] = type_map_special_default[f]
            # sentence
            elif sh_inferred_type == 3:
                type_map_special[f] = S_TEXT
            elif f in type_map_special_default:
                type_map_special[f] = type_map_special_default[f]
        return type_map_special

    def _get_type_map_sh(self, features, y_RF):
        return {features[i]: y_RF[i] for i in range(len(features))}
