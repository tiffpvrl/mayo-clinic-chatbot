from datetime import datetime, timedelta, timezone

def build_demo_schedule(record: dict, days_until_colonoscopy: int = 3) -> dict:
    original_col = record.get("colonoscopy_datetime")
    original_prep_start = record.get("bowel_prep_start_datetime")
    original_prep_end = record.get("bowel_prep_end_datetime")

    if not original_col or not original_prep_start or not original_prep_end:
        raise ValueError("Missing procedure timeline fields.")

    now = datetime.now(timezone.utc)
    demo_col = now + timedelta(days=days_until_colonoscopy)

    prep_start_delta = original_col - original_prep_start
    prep_end_delta = original_col - original_prep_end

    demo_prep_start = demo_col - prep_start_delta
    demo_prep_end = demo_col - prep_end_delta

    return {
        "demo_colonoscopy_datetime": demo_col,
        "demo_bowel_prep_start_datetime": demo_prep_start,
        "demo_bowel_prep_end_datetime": demo_prep_end,
    }