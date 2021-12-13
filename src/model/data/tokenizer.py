import re

import spacy
from spacy.tokens.token import Token


class TokenizerCPN:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, text):
        all_tokens_data = []
        doc = self.nlp(text)
        tok_id = 0

        for tok in doc:
            tokp: Token = tok

            found_year = re.findall('([1-3][0-9]{3})', tokp.text)
            found_s_end_date = re.findall('([0-9]{1,4}s)', tokp.text)

            found_dot_after_month = re.findall('(?i)((january|'
                                               'february|'
                                               'march|'
                                               'april|may|june|july|august|september|october|'
                                               'november|december)\\.)', tokp.text)

            # this particular case has to be hardcoded (ex: "2012/2013"), if not just taken as single token
            # something that produces inconsistencies later in dataset (one token pointing to two concepts).
            if len(found_year) == 2 and len(str(tokp)) == 9 and '/' in str(tokp) and str(tokp).index('/') == 4:
                # print('processing dates separated by slash case: ', str(tokp))
                tok1 = tokp.text[0:4]
                tok2 = tokp.text[4:5]
                tok3 = tokp.text[5:9]
                all_tokens_data.append({'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + 4,
                                        'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + 4, 'tokid_end_char': tokp.idx + 5,
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + 5, 'tokid_end_char': tokp.idx + 9,
                                        'tok_id': tok_id, 'tok_text': tok3})
                tok_id += 1
            # this particular case happens when date (year for example) is finished in 's', ex: 1980s, 90s, etc.
            # it has to be parsed as two different tokens (ex: ['1980', 's']), with spacy version it gets parsed
            # as single token which produces difference in scores with Johannes version later on
            elif len(found_s_end_date) == 1 and len(found_s_end_date[0]) == len(str(tokp)):
                # print('processing date ending in s: ', str(tokp))
                tok1 = str(tokp)[0:len(str(tokp)) - 1]
                tok2 = str(tokp)[len(str(tokp)) - 1: len(str(tokp))]
                all_tokens_data.append(
                    {'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(str(tokp)) - 1,
                     'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + len(str(tokp)) - 1,
                                        'tokid_end_char': tokp.idx + len(str(tokp)),
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1
            # this particular case happens when there is a dot after month, sometimes spacy doesn't separate the
            # dot from the month, ex 'May.' gets parsed as ['May.'] and not ['May','.']
            elif len(found_dot_after_month) == 1 and len(found_dot_after_month[0][0]) == len(str(tokp)):
                # print('processing month name ending in .', str(tokp))
                tok1 = str(tokp)[0:len(str(tokp)) - 1]
                tok2 = str(tokp)[len(str(tokp)) - 1: len(str(tokp))]
                all_tokens_data.append({'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(str(tokp)) - 1,
                                        'tok_id': tok_id, 'tok_text': tok1})
                tok_id += 1
                all_tokens_data.append({'tokid_begin_char': tokp.idx + len(str(tokp)) - 1,
                                        'tokid_end_char': tokp.idx + len(str(tokp)),
                                        'tok_id': tok_id, 'tok_text': tok2})
                tok_id += 1

            else:
                token_data = {'tokid_begin_char': tokp.idx, 'tokid_end_char': tokp.idx + len(tokp.text),
                              'tok_id': tok_id, 'tok_text': tokp.text}
                all_tokens_data.append(token_data)
                tok_id += 1
        return [{'offset': tok_info['tokid_begin_char'],
                 'length': tok_info['tokid_end_char'] - tok_info['tokid_begin_char'],
                 'token': tok_info['tok_text']} for tok_info in all_tokens_data]
