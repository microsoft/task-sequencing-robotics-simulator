# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import typing
from enum import Enum


class ConceptType(Enum):
    CONCEPT_GRASP_ACTIVE_FORCE = 0 
    CONCEPT_RELEASE = 999

    CONCEPT_PTG11 = 1110  # also known as pick  
    CONCEPT_PTG12 = 1120  # also known as bring with position-only goals
    CONCEPT_PTG13 = 1132  # also known as place for release

    CONCEPT_EXECUTE = 9900

def ToStr(concepts: typing.List[ConceptType]) -> str:
    ret = ""
    for concept in concepts:
        if concept.value < 10: ret += "000" + str(concept.value)
        elif concept.value < 100: ret += "00" + str(concept.value)
        elif concept.value < 1000: ret += "0" + str(concept.value)
        else: ret += str(concept.value)
    return ret
