
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
    # http://shakespeare-online.com/keydates/playchron.html
    return \
    {
        "All's Well That Ends Well" : '1602-03 / 1623',
        'As You Like It' : '1599-1600 / 1623',
        'The Comedy of Errors' : '1592-93 / 1623',
        'Cymbeline' : '1609-10 / 1623',
        "Love's Labours Lost" : '1594-95 / 1598?',
        'Measure for Measure' : '1604-05 / 1623',
        'The Merry Wives of Windsor' : '1600-01 / 1602',
        'The Merchant of Venice' : '1596-97 / 1600',
        "A Midsummer Night's Dream" : '1595-96 / 1600',
        'Much Ado About Nothing' : '1598-99 / 1600',
        'Pericles, Prince of Tyre' : '1608-09 / 1609',
        'Taming of the Shrew' : '1593-94 / 1623',
        'The Tempest' : '1611-12 / 1623',
        'Troilus and Cressida' : '1601-02 / 1609',
        'Twelfth Night' : '1599-1600 / 1623',
        'Two Gentlemen of Verona' : '1594-95 / 1623',
        "Winter's Tale" : '1610-11 / 1623',
        'Henry IV, part 1' : '1597-98 / 1598',
        'Henry IV, part 2' : '1597-98 / 1600',
        'Henry V' : '1598-99 / 1600',
        'Henry VI, part 1' : '1591-92 / 1623',
        'Henry VI, part 2' : '1590-91 / 1594?',
        'Henry VI, part 3' : '1590-91 / 1594?',
        'Henry VIII' : '1612-13 / 1623',
        'King John' : '1596-97 / 1623',
        'Richard II' : '1595-96 / 1597',
        'Richard III' : '1592-93 / 1597',
        'Antony and Cleopatra' : '1606-07 / 1623',
        'Coriolanus' : '1607-08 / 1623',
        'Hamlet' : '1600-01 / 1603',
        'Julius Caesar' : '1599-1600 / 1623',
        'King Lear' : '1605-06 / 1608',
        'Macbeth' : '1605-06 / 1623',
        'Othello' : '1604-05 / 1622',
        'Romeo and Juliet' : '1594-95 / 1597',
        'Timon of Athens' : '1607-08 / 1623',
        'Titus Andronicus' : '1593-94 / 1594',
    }

classifications = _get_classifications()
vintage = _get_years()
