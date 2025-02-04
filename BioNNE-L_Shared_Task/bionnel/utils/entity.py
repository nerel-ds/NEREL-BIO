class Entity:
    def __init__(self, doc_id, e_id, spans, e_type, entity_str, cui=None):
        self.doc_id = doc_id
        self.e_id = e_id
        self.spans = spans
        self.e_type = e_type
        self.entity_str = entity_str
        self.cui = cui

        min_span = min((int(t[0]) for t in self.spans))
        max_span = max((int(t[1]) for t in self.spans))
        self.min_span = min_span
        self.max_span = max_span

    def __str__(self):
        # min_span = min((int(t[0]) for t in self.spans))
        # max_span = max((int(t[1]) for t in self.spans))

        return f"{self.doc_id}|{self.e_id}||{str(self.spans)}||{self.e_type}||{self.entity_str}||{self.cui}"


    def as_dict(self):
        spans_s = ','.join((f"{t[0]}-{t[1]}" for t in self.spans))
        d = {
            "document_id": self.doc_id,
            "text": self.entity_str,
            "entity_type": self.e_type,
            "spans": spans_s,
            "UMLS_CUI": self.cui
        }
        return d

    def to_biosyn_str(self):
        assert '||' not in self.e_id
        assert '||' not in self.e_type
        assert '||' not in self.entity_str

        return self.__str__()
