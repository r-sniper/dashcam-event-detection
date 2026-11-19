from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y:%m:%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
