"""Declaration and definition of exceptions that can be raised from user_recommendation module"""
from enum import IntEnum, auto
from typing import Optional


class UserRecommendationErrorCode(IntEnum):
    """Error identifiers of user recommendation's errors

    Attributes:\
        MISSING_ATTRIBUTE: see MissingAttribute
        INPUT_DATA_ERROR: see
        BAD_EVALUATION_METRIC: see
        INVALID_TAG: see InvalidTag
    """

    UNKNOWN = -1
    MISSING_ATTRIBUTE = auto()
    INPUT_DATA_ERROR = auto()
    BAD_EVALUATION_METRIC = auto()
    INVALID_TAG = auto()


class UserRecommendationException(Exception):
    """Exception raised from the module user_recommendation

    Args:
        code: error code identifier, default to -1

    Attributes
        _code: numeric value associated with an error

    """

    def __init__(self, code: UserRecommendationErrorCode = UserRecommendationErrorCode.UNKNOWN) -> None:
        """Call parent class init and set error code to unknown value (-1).

        Args:
            code: see _code attributes docstring.

        """
        super().__init__()
        self._code = code

    @property
    def code(self) -> UserRecommendationErrorCode:
        """Get _code attribute.

        Returns
            CacaoErrorCode: See _code attributes docstring.

        """
        return self._code

    def __str__(self) -> str:
        """Overridden __str__ method.

        Returns
        -------
            str: Formatted error message.

        """
        return f"{self.__class__.__name__}|{self._code}|"


class MissingAttribute(UserRecommendationException):
    """Exception raised if we want to acces to a missing attribute

    Attributes
        _attribute: identifier that is unknown
        _valid_attributes: valid attributes formatted as 'attr1, attr2, ..'

    """

    _valid_attributes: Optional[str]

    def __init__(self, attribute: str, valid_attributes: Optional[list[str]] = None) -> None:
        """Initialize error.

        Args:
            attribute: see _attribute attributes docstring.
            valid_attributes: list of valid attributes

        """
        super().__init__(UserRecommendationErrorCode.MISSING_ATTRIBUTE)
        self._attribute = attribute
        self._valid_attributes = ", ".join(valid_attributes) if isinstance(valid_attributes, list) else None

    def __str__(self) -> str:
        """Overridden __str__ method.

        Returns
            str: Formatted error message.

        """
        suffix = f" Use one of the following:\n{self._valid_attributes}" if self._valid_attributes else ""
        return super().__str__() + f"{self._attribute} is unknown.{suffix}"


class InputDataError(UserRecommendationException):
    """Exception raised if error when download or load input data"""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__(UserRecommendationErrorCode.INPUT_DATA_ERROR)

    def __str__(self) -> str:
        """Overridden __str__ method.

        Returns
            str: Formatted error message.

        """
        return super().__str__() + "Error handling input data"


class BadEvaluationMetric(UserRecommendationException):
    """Exception raised a bad evaluation metric is called during training phase

    Attributes:

        _available_metric_str: Available metrics to called format as: metr1, metr2, ...
    """

    def __init__(self, metric: str, available_metrics: list[str]) -> None:
        """Error constructor

        Args:
            metric: Wrong metric called
            available_metrics: List of callable metrics
        """
        super().__init__(UserRecommendationErrorCode.BAD_EVALUATION_METRIC)
        self._metric = metric
        self._available_metric_str = ", ".join(available_metrics)

    def __str__(self) -> str:
        """Overridden __str__ method.

        Returns
            str: Formatted error message.

        """
        return (
            super().__str__()
            + f"""Bad metric called '{self._metric}', please choose one of these
                                        following evaluation metrics:\n
                                        {self._available_metric_str}"""
        )


class InvalidTag(UserRecommendationException):
    """Exception raised when the input tag is not valid

    Attributes
        _bad_name: identifier that is unknown
        _doc: documentation of the enumerator where the identifiers are defined

    """

    _bad_name: str
    _doc: str

    def __init__(self, bad_name: str, enum_doc: str) -> None:
        """Initialize error.

        Args:
            bad_name: see _bad_name attributes docstring.
            enum_doc: see _doc attributes docstring.

        """
        super().__init__(UserRecommendationErrorCode.INVALID_TAG)
        self._bad_name = bad_name
        self._doc = enum_doc

    def __str__(self) -> str:
        """Overridden __str__ method.

        Returns
            str: Formatted error message.

        """
        return f"Tag {self._bad_name} unknown, use one of the following identifiers:\n{self._doc}"
