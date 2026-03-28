from enum import Enum


class TrainingStatus(str, Enum):
    SUCCESS = 'success' 
    NO_ENV = 'no environment'
    NO_MODEL = 'no model'
    ERROR = 'error'
