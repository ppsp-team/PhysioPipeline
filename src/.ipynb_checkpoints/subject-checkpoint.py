
class Subject:

    id: int
    role_id: int
    role_desc: str

    def __init__(self, id: int, role_id: int, role_desc: str) -> None:
        self.id = id
        self.role_id = role_id
        self.role_desc = role_desc

