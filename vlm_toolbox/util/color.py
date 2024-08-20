import distinctipy


def convert_float_tuple_to_rgb_string(rgb_tuple):
    r, g, b = [int(255 * x) for x in rgb_tuple]
    return f"rgb({r},{g},{b})"

def convert_rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_diverse_colors(count, return_str=True, color_format='rgb'):
    if not count or count < 0:
        return []

    colors = distinctipy.get_colors(count)
    if return_str:
        if color_format == 'rgb':
            colors = [convert_float_tuple_to_rgb_string(c) for c in colors]
        elif color_format == 'hex':
            colors = [convert_rgb_to_hex(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in colors]
    return colors
