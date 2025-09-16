


def convert_time(start,end):
    delta = end - start
    minutes = round(delta // 60)
    delta %= 60
    seconds = round(delta)
    time_str = f"{minutes:02d}m{seconds:02d}s"
    return time_str