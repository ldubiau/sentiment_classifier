from classifiers.processors import FreeLingProcessor


def main():
    processor = FreeLingProcessor()
    processed = processor.process([['hola', 'mundo', 'lindo'], ['chau', 'mundo', 'feo']])
    print processed
    processor = FreeLingProcessor(lambda term: term.tag.startswith('AQ'))
    processed = processor.process([['hola', 'mundo', 'lindo'], ['chau', 'mundo', 'feo']])
    print processed
    processor = FreeLingProcessor(lambda term: term.tag.startswith('NC'))
    processed = processor.process([['hola', 'mundo', 'lindo'], ['chau', 'mundo', 'feo']])
    print processed


if __name__ == '__main__':
    main()
