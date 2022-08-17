class AntescofoFormatting:
    @staticmethod
    def write_list_as_antescofo_map(seq, start_idx):
        s = "MAP {"
        i = 0

        for el in seq:
            s += "({}, {})".format(i + start_idx, AntescofoFormatting.write_obj_as_antescofo_struct(el))
            i += 1
            if i < len(seq):
                s += ", "
            else:
                s += " }"
        return s

    @staticmethod
    def write_obj_as_antescofo_struct(obj):
        if len(obj) == 2:
            return AntescofoFormatting.write_content_and_tranfo_as_antescofo_tab(obj[0] - 1, obj[1])

    @staticmethod
    def write_content_and_tranfo_as_antescofo_tab(content, transfo):
        return "TAB [{}, {}]".format(content, transfo)
