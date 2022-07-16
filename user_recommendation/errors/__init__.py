"""Declaration and definition of exceptions that can be raised from user_recommendation module"""
from enum import IntEnum, auto
from typing import Optional


class UserRecommendationErrorCode(IntEnum):
    """Error identifiers of user recommendation's errors

    Attributes:\
        MissingAttribute: see MissingAttribute
    """

    UNKNOWN = -1
    MISSING_ATTRIBUTE = auto()


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
        return f"{self._attribute} is unknown.{suffix}"
