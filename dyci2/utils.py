from model_navigator import FactorOracleGenerator

IntQ = int


def format_list_as_list_of_strings(l):
    result = []
    for i in l:
        i_s = ""
        if type(i) == list:
            for j in range(len(i) - 1):
                i_s += format_obj_as_string(i[j]) + " "
            i_s += format_obj_as_string(i[len(i) - 1])
        else:
            i_s = format_obj_as_string(i)
        result.append(i_s)

    return result


def format_obj_as_string(o):
    s = ""
    if type(o) == str:
        s = o.replace("u'", "").replace("'", "")
    else:
        s = format(o)
    return s


implemented_model_navigator_classes = {"FactorOracleNavigator": FactorOracleGenerator}


def noneIsInfinite(value):
    if value is None:
        return float("inf")
    else:
        return value