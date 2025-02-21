from pydantic import BaseModel


class Venue(BaseModel):
    """
    Represents the data structure of a Venue.
    """

    document_name: str
    document_url: str
    error: bool = False
