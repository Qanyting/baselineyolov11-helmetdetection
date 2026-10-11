from time import perf_counter


def process_image(detector, source, destination, conf):
    started = perf_counter()
    result = detector.detect_image(source, destination, conf)
    result["processing_time"] = perf_counter() - started
    return result
