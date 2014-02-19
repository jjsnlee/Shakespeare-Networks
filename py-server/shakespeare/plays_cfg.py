
plays = [
    # Will use this as the key/value pair
    ('allswell', "All's Well That Ends Well"),
    ('asyoulikeit', 'As You Like It'),
    ('comedy_errors', 'The Comedy of Errors'),
    ('cymbeline', 'Cymbeline'),
    ('lll', "Love's Labours Lost"),
    ('measure', 'Measure for Measure'),
    ('merry_wives', 'The Merry Wives of Windsor'),
    ('merchant', 'The Merchant of Venice'),
    ('midsummer', "A Midsummer Night's Dream"),
    ('much_ado', 'Much Ado About Nothing'),
    ('pericles', 'Pericles, Prince of Tyre'),
    ('taming_shrew', 'Taming of the Shrew'),
    ('tempest', 'The Tempest'),
    ('troilus_cressida', 'Troilus and Cressida'),
    ('twelfth_night', 'Twelfth Night'),
    ('two_gentlemen', 'Two Gentlemen of Verona'),
    ('winters_tale', "Winter's Tale"),
    ('1henryiv', 'Henry IV, part 1'),
    ('2henryiv', 'Henry IV, part 2'),
    ('henryv', 'Henry V'),
    ('1henryvi', 'Henry VI, part 1'),
    ('2henryvi', 'Henry VI, part 2'),
    ('3henryvi', 'Henry VI, part 3'),
    ('henryviii', 'Henry VIII'),
    ('john', 'King John'),
    ('richardii', 'Richard II'),
    ('richardiii', 'Richard III'),
    ('cleopatra', 'Antony and Cleopatra'),
    ('coriolanus', 'Coriolanus'),
    ('hamlet', 'Hamlet'),
    ('julius_caesar', 'Julius Caesar'),
    ('lear', 'King Lear'),
    ('macbeth', 'Macbeth'),
    ('othello', 'Othello'),
    ('romeo_juliet', 'Romeo and Juliet'),
    ('timon', 'Timon of Athens'),
    ('titus', 'Titus Andronicus')
]

def _get_classifications():
    _comedies = dict.fromkeys([
                    "All's Well That Ends Well",
                    'As You Like It',
                    'The Comedy of Errors',
                    'Cymbeline',
                    "Love's Labours Lost",
                    'Measure for Measure',
                    'The Merry Wives of Windsor',
                    'The Merchant of Venice',
                    "A Midsummer Night's Dream",
                    'Much Ado About Nothing',
                    'Pericles, Prince of Tyre',
                    'Taming of the Shrew',
                    'The Tempest',
                    'Troilus and Cressida',
                    'Twelfth Night',
                    'Two Gentlemen of Verona',
                    "Winter's Tale"
                  ], 
                  'Comedy')
    
    _hists = dict.fromkeys([
                    'Henry IV, part 1',
                    'Henry IV, part 2',
                    'Henry V',
                    'Henry VI, part 1',
                    'Henry VI, part 2',
                    'Henry VI, part 3',
                    'Henry VIII',
                    'King John',
                    'Richard II',
                    'Richard III'
                    ], 'History')
    _tragedies = dict.fromkeys([
                    'Antony and Cleopatra',
                    'Coriolanus',
                    'Hamlet',
                    'Julius Caesar',
                    'King Lear',
                    'Macbeth',
                    'Othello',
                    'Romeo and Juliet',
                    'Timon of Athens',
                    'Titus Andronicus',
                     ], 'Tragedy'
                     )
    
    play_classifications = _comedies
    play_classifications.update(_hists)
    play_classifications.update(_tragedies)
    return play_classifications

def _get_years():
    return \
    {
        "All's Well That Ends Well" : 1603,
        'As You Like It' : 1600,
        'The Comedy of Errors' : 1593,
        'Cymbeline' : 1610,
        "Love's Labours Lost" : 1595,
        'Measure for Measure' : 1605,
        'The Merry Wives of Windsor' : 1601,
        'The Merchant of Venice' : 1597,
        "A Midsummer Night's Dream" : 1596,
        'Much Ado About Nothing' : 1599,
        'Pericles, Prince of Tyre' : 1609,
        'Taming of the Shrew' : 1594,
        'The Tempest' : 1612,
        'Troilus and Cressida' : 1602,
        'Twelfth Night' : 1600,
        'Two Gentlemen of Verona' : 1595,
        "Winter's Tale" : 1611,
        'Henry IV, part 1' : 1598,
        'Henry IV, part 2' : 1598,
        'Henry V' : 1599,
        'Henry VI, part 1' : 1592,
        'Henry VI, part 2' : 1591,
        'Henry VI, part 3' : 1591,
        'Henry VIII' : 1613,
        'King John' : 1597,
        'Richard II' : 1596,
        'Richard III' : 1593,
        'Antony and Cleopatra' : 1608,
        'Coriolanus' : 1608,
        'Hamlet' : 1601,
        'Julius Caesar' : 1600,
        'King Lear' : 1606,
        'Macbeth' : 1606,
        'Othello' : 1605,
        'Romeo and Juliet' : 1595,
        'Timon of Athens' : 1608,
        'Titus Andronicus' : 1594,
    }

classifications = _get_classifications()
vintage = _get_years()
