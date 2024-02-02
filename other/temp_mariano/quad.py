points = [[0.0571041961,  0.06546699455602246],
            [0.2768430136,  0.05021012321401679],
            [0.5835904324,  0.02891208422223085],
            [0.8602401357,  0.009703785123906346],
            [0.0571041961,  0.3111645522491480],
            [0.2768430136,  0.2386486597440242],
            [0.5835904324,  0.1374191041243166],
            [0.8602401357,  0.04612207989200404],
            [0.0571041961,  0.6317312516508520],
            [0.2768430136,  0.4845083266559759],
            [0.5835904324,  0.2789904634756834],
            [0.8602401357,  0.09363778440799593],
            [0.0571041961,  0.8774288093439775],
            [0.2768430136,  0.6729468631859832],
            [0.5835904324,  0.3874974833777692],
            [0.8602401357,  0.1300560791760936]]

weights = [0.04713673637581137,
            0.07077613579259895,
            0.04516809856187617,
            0.01084645180365496,
            0.08837017702418863,
            0.1326884322074010,
            0.08467944903812383,
            0.02033451909634504,
            0.08837017702418863,
            0.1326884322074010,
            0.08467944903812383,
            0.02033451909634504,
            0.04713673637581137,
            0.07077613579259895,
            0.04516809856187617,
            0.01084645180365496]

# points =  [[   0.9553660447100000     ,  0.8862103848242247E-03],
#           [  0.9553660447100000      , 0.4537789678039195E-02],
#           [  0.9553660447100000      , 0.1058868260117431E-01],
#           [  0.9553660447100000      , 0.1822327082910602E-01],
#           [  0.9553660447100000      , 0.2641068446089399E-01],
#           [  0.9553660447100000      , 0.3404527268882569E-01],
#           [  0.9553660447100000      , 0.4009616561196080E-01],
#           [  0.9553660447100000      , 0.4374774490517578E-01],
#           [  0.8556337429600001      , 0.2866402391985981E-02],
#           [  0.8556337429600001      , 0.1467724979327651E-01],
#           [  0.8556337429600001      , 0.3424855503358430E-01],
#           [  0.8556337429600001      , 0.5894224214571626E-01],
#           [  0.8556337429600001      , 0.8542401489428375E-01],
#           [  0.8556337429600001      , 0.1101177020064157    ],
#           [  0.8556337429600001      , 0.1296890072467235    ],
#           [  0.8556337429600001      , 0.1414998546480140    ],
#           [  0.7131752428600000      , 0.5694926133044352E-02],
#           [  0.7131752428600000      , 0.2916054411712861E-01],
#           [  0.7131752428600000      , 0.6804452564827500E-01],
#           [  0.7131752428600000      , 0.1171055801775613    ],
#           [  0.7131752428600000      , 0.1697191769624387    ],
#           [  0.7131752428600000      , 0.2187802314917250    ],
#           [  0.7131752428600000      , 0.2576642130228714    ],
#           [  0.7131752428600000      , 0.2811298310069557    ],
#           [  0.5451866848000000      , 0.9030351006711630E-02],
#           [  0.5451866848000000      , 0.4623939674940125E-01],
#           [  0.5451866848000000      , 0.1078970888004545    ],
#           [  0.5451866848000000      , 0.1856923986620134    ],
#           [  0.5451866848000000      , 0.2691209165379867    ],
#           [  0.5451866848000000      , 0.3469162263995455    ],
#           [  0.5451866848000000      , 0.4085739184505988    ],
#           [  0.5451866848000000      , 0.4457829641932884    ],
#           [  0.3719321645800000      , 0.1247033193690498E-01],
#           [  0.3719321645800000      , 0.6385362269957356E-01],
#           [  0.3719321645800000      , 0.1489989161403976    ],
#           [  0.3719321645800000      , 0.2564292182833579    ],
#           [  0.3719321645800000      , 0.3716386171366422    ],
#           [  0.3719321645800000      , 0.4790689192796024    ],
#           [  0.3719321645800000      , 0.5642142127204264    ],
#           [  0.3719321645800000      , 0.6155975034830951    ],
#           [  0.2143084794000000      , 0.1559996151584746E-01],
#           [  0.2143084794000000      , 0.7987871227492103E-01],
#           [  0.2143084794000000      , 0.1863925811641285    ],
#           [  0.2143084794000000      , 0.3207842387034378    ],
#           [  0.2143084794000000      , 0.4649072818965623    ],
#           [  0.2143084794000000      , 0.5992989394358715    ],
#           [  0.2143084794000000      , 0.7058128083250790    ],
#           [  0.2143084794000000      , 0.7700915590841526    ],
#           [  0.9132360790000005E-01  , 0.1804183496379599E-01],
#           [  0.9132360790000005E-01  , 0.9238218584838476E-01],
#           [  0.9132360790000005E-01  , 0.2155687489628060    ],
#           [  0.9132360790000005E-01  , 0.3709968314854498    ],
#           [  0.9132360790000005E-01  , 0.5376795606145502    ],
#           [  0.9132360790000005E-01  , 0.6931076431371940    ],
#           [  0.9132360790000005E-01  , 0.8162942062516152    ],
#           [  0.9132360790000005E-01  , 0.8906345571362040    ],
#           [  0.1777991514999999E-01  , 0.1950205026019779E-01],
#           [  0.1777991514999999E-01  , 0.9985913490381848E-01],
#           [  0.1777991514999999E-01  , 0.2330157982952792    ],
#           [  0.1777991514999999E-01  , 0.4010234473667467    ],
#           [  0.1777991514999999E-01  , 0.5811966374832533    ],
#           [  0.1777991514999999E-01  , 0.7492042865547208    ],
#           [  0.1777991514999999E-01  , 0.8823609499461815    ],
#           [  0.1777991514999999E-01  , 0.9627180345898023    ]]


# weights = [    0.3335674062677772E-03,
#     0.7327880811491046E-03,
#     0.1033723454167925E-02,
#     0.1195112498415193E-02,
#     0.1195112498415193E-02,
#     0.1033723454167925E-02,
#     0.7327880811491046E-03,
#     0.3335674062677772E-03,
#     0.1806210919443461E-02,
#     0.3967923151181667E-02,
#     0.5597437146194232E-02,
#     0.6471331443180639E-02,
#     0.6471331443180639E-02,
#     0.5597437146194232E-02,
#     0.3967923151181667E-02,
#     0.1806210919443461E-02,
#     0.4599755803015752E-02,
#     0.1010484287526739E-01,
#     0.1425461651131868E-01,
#     0.1648010431039818E-01,
#     0.1648010431039818E-01,
#     0.1425461651131868E-01,
#     0.1010484287526739E-01,
#     0.4599755803015752E-02,
#     0.8017259531156730E-02,
#     0.1761248886287915E-01,
#     0.2484544071087993E-01,
#     0.2872441038508419E-01,
#     0.2872441038508419E-01,
#     0.2484544071087993E-01,
#     0.1761248886287915E-01,
#     0.8017259531156730E-02,
#     0.1073501897357062E-01,
#     0.2358292149331603E-01,
#     0.3326776143412911E-01,
#     0.3846165753898425E-01,
#     0.3846165753898425E-01,
#     0.3326776143412911E-01,
#     0.2358292149331603E-01,
#     0.1073501897357062E-01,
#     0.1138879740452669E-01,
#     0.2501915606814251E-01,
#     0.3529381699354388E-01,
#     0.4080402900378691E-01,
#     0.4080402900378691E-01,
#     0.3529381699354388E-01,
#     0.2501915606814251E-01,
#     0.1138879740452669E-01,
#     0.9223845391285393E-02,
#     0.2026314273544469E-01,
#     0.2858464328177232E-01,
#     0.3304739223149761E-01,
#     0.3304739223149761E-01,
#     0.2858464328177232E-01,
#     0.2026314273544469E-01,
#     0.9223845391285393E-02,
#     0.4509812715921713E-02,
#     0.9907253959306707E-02,
#     0.1397588340693756E-01,
#     0.1615785427783403E-01,
#     0.1615785427783403E-01,
#     0.1397588340693756E-01,
#     0.9907253959306707E-02,
#     0.4509812715921713E-02]