def convert_to_json(identifier, content, begin, end, ner, coref, rels, mention_rels, singletons=False):
    builder = BuilderDoc(content, begin, end)
    builder.set_id(identifier)
    builder.singletons = singletons

    for begin, end, tag in ner:
        mention = builder.add_mention(begin, end - 1)  # exclusive
        mention.add_tag(tag)

    if coref is not None:
        for cluster in coref:
            concept = builder.add_concept()
            for begin, end in cluster:
                concept.add_mention(begin, end)

    if rels is not None:
        for src_cluster, dst_cluster, rel in rels:
            src = builder.get_concept(src_cluster)
            dst = builder.get_concept(dst_cluster)
            builder.add_relation(src, dst, rel)

    if mention_rels is not None:
        for src_mention, rel, dst_mention in mention_rels:
            src = builder.add_mention(src_mention[0], src_mention[1])
            dst = builder.add_mention(dst_mention[0], dst_mention[1])
            builder.add_mention_relation(src, dst, rel)

    return builder.json()


class BuilderDoc:

    def __init__(self, content, begin, end):
        self.content = content
        self.begin = begin
        self.end = end
        self.identifier = None
        self.mentions = []
        self.concepts = []
        self.relations = []
        self.mention_relations = []
        self.span2mention = {}

    def set_id(self, identifier):
        self.identifier = identifier

    def add_concept(self):
        concept = BuilderConcept(self)
        self.concepts.append(concept)
        return concept

    def add_mention(self, begin, end):
        span = (begin, end)
        if span not in self.span2mention:
            mention = BuilderMention(self, begin, end)
            self.mentions.append(mention)
            self.span2mention[span] = mention
        return self.span2mention[span]

    def add_relation(self, src, dst, rel):
        relation = BuilderRelation(src, dst, rel)
        self.relations.append(relation)
        src.add_relation(relation)
        dst.add_relation(relation)

    def add_mention_relation(self, src, dst, rel):
        relation = BuilderRelation(src, dst, rel)
        self.mention_relations.append(relation)
        src.add_mention_relation(relation)
        dst.add_mention_relation(relation)

    def get_mention(self, begin, end):
        for m in self.mentions:
            if m.token_begin == begin and m.token_end == end:
                return m
        return None

    def get_concept(self, cluster):
        concepts = []
        for begin, end in cluster:
            mention = self.get_mention(begin, end)
            concepts.append(mention.concept)
        for c in concepts:
            if c != concepts[0]:
                print('RELATION HAS MULTIPLE CONCEPTS IN CLUSTER')
        return concepts[0]

    def json(self):
        # create concepts for mentions without one
        for mention in self.mentions:
            if mention.concept is None:
                self.add_concept().add_mention2(mention)

        # number concepts
        idx = 0
        for concept in self.concepts:
            concept._visible = concept.is_visible()
            if concept._visible:
                concept.idx = idx
                idx += 1

        # number mentions
        idx = 0
        for mention in self.mentions:
            mention._visible = mention.concept._visible or len(mention.mention_relations) > 0
            if mention._visible:
                mention.idx = idx
                idx += 1

        return {
            'id': self.identifier,
            'content': self.content,
            'mentions': [m.json() for m in self.mentions if m._visible],
            'concepts': [c.json() for c in self.concepts if c._visible],
            'relations': [r.json() for r in self.relations],
            'mention_relations': [r.json() for r in self.mention_relations],
            'frames': []
        }


class BuilderConcept:

    def __init__(self, doc):
        self.doc = doc
        self.idx = -1
        self.mentions = []
        self.relations = []

    def add_mention(self, begin, end):
        mention = self.doc.add_mention(begin, end)
        mention.concept = self
        self.mentions.append(mention)

    def add_mention2(self, mention):
        mention.concept = self
        self.mentions.append(mention)

    def add_relation(self, relation):
        self.relations.append(relation)

    def is_visible(self):
        return len(self.get_tags()) > 0 or len(self.mentions) > 1 or len(self.relations) > 0 or self.doc.singletons

    def get_text(self):
        text = None
        for mention in self.mentions:
            text = mention.text if text is None or len(text) < len(mention.text) else text
        return text

    def get_tags(self):
        tags = set()
        for mention in self.mentions:
            tags.update(mention.tags)
        return tags

    def json(self):
        return {
            'concept': self.idx,
            'text': self.get_text(),
            'count': len(self.mentions),
            'tags': list(self.get_tags())
        }


class BuilderMention:

    def __init__(self, doc, token_begin, token_end):
        self.tags = set()
        self.token_begin = token_begin
        self.token_end = token_end
        self.char_begin = doc.begin[token_begin]
        self.char_end = doc.end[token_end]
        self.text = doc.content[self.char_begin:self.char_end]
        self.concept = None
        self.mention_relations = []
        self.idx = -1

    def add_tag(self, tag):
        self.tags.add(tag)

    def add_mention_relation(self, relation):
        self.mention_relations.append(relation)

    def json(self):
        return {
            'concept': self.concept.idx,
            'begin': self.char_begin,
            'end': self.char_end,
            'text': self.text,
            'tags': list(self.tags)
        }


class BuilderRelation:

    def __init__(self, src, dst, rel):
        self.src = src
        self.dst = dst
        self.rel = rel

    def json(self):
        return {
            's': self.src.idx,
            'p': self.rel,
            'o': self.dst.idx
        }
