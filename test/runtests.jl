using MusicViz

using Bplus; @using_bplus


for test in [ [ 3, 5, 8, 20 ],
              [ 10, -20, 30, -100 ],
              (0, 0),
              ( 1, 3, 6, -1),
              (1, ),
              () ]
    @bp_check(collect(iter_diff(test)) ==
                diff(collect(test)),
              test, " => ", collect(iter_diff(test)))
    @bp_check(collect(iter_cumsum(test)) ==
                cumsum(collect(test)),
              test, " => ", collect(iter_cumsum(test)))
end