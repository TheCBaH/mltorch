Run efficientnet_b0 (the smallest/fastest efficientnet variant) end-to-end
through the graph interpreter on every sample image and compare the local
top-5 against the reference results.json shipped in the release zip, in
--cram mode: when the local top-5 ranking matches the reference, only the
ranking prints (probabilities differ in their low-order digits across
systems, so they'd otherwise make this comparison non-deterministic); on a
mismatch both full rankings print with probabilities at natural precision
for inspection. Per-image inference time goes to stderr, so it is dropped
here too; see `make inference` to see timing and natural-precision
probabilities for any downloaded model. Gated on PT2_DATA so the default
suite needs no download; run via `make pt2.runtest` after
`make pt2.download PT2_MODEL=efficientnet_b0`.

  $ ./interp_run.exe "$PT2_DATA/efficientnet_b0/efficientnet_b0.pt2" "$PT2_DATA/efficientnet_b0/images" \
  >   "$PT2_DATA/efficientnet_b0/imagenet_lsvrc_2015_synsets.txt" \
  >   "$PT2_DATA/efficientnet_b0/imagenet_metadata.txt" "$PT2_DATA/efficientnet_b0/results.json" --cram 2>/dev/null
  === images/000000000149.pt ===
  1: n03888257 parachute, chute
  2: n02692877 airship, dirigible
  3: n03355925 flagpole, flagstaff
  4: n02782093 balloon
  5: n03956157 planetarium
  === images/000000000201.pt ===
  1: n04228054 ski
  2: n03218198 dogsled, dog sled, dog sleigh
  3: n04208210 shovel
  4: n04252077 snowmobile
  5: n02860847 bobsled, bobsleigh, bob
  === images/000000000349.pt ===
  1: n03272562 electric locomotive
  2: n03895866 passenger car, coach, carriage
  3: n03393912 freight car
  4: n04310018 steam locomotive
  5: n02917067 bullet train, bullet
  === images/000000000389.pt ===
  1: n03630383 lab coat, laboratory coat
  2: n02883205 bow tie, bow-tie, bowtie
  3: n04591157 Windsor tie
  4: n04532106 vestment
  5: n04317175 stethoscope
  === images/000000000404.pt ===
  1: n03216828 dock, dockage, docking facility
  2: n09332890 lakeside, lakeshore
  3: n03662601 lifeboat
  4: n02859443 boathouse
  5: n04612504 yawl
  === images/000000000438.pt ===
  1: n07695742 pretzel
  2: n02776631 bakery, bakeshop, bakehouse
  3: n07697537 hotdog, hot dog, red hot
  4: n07693725 bagel, beigel
  5: n04476259 tray
  === images/000000000564.pt ===
  1: n03032252 cinema, movie theater, movie theatre, movie house, picture palace
  2: n04296562 stage
  3: n04418357 theater curtain, theatre curtain
  4: n02802426 basketball
  5: n04149813 scoreboard
  === images/000000000599.pt ===
  1: n04074963 remote control, remote
  2: n02123045 tabby, tabby cat
  3: n02123159 tiger cat
  4: n02124075 Egyptian cat
  5: n03642806 laptop, laptop computer
  === images/000000000605.pt ===
  1: n07920052 espresso
  2: n07930864 cup
  3: n07579787 plate
  4: n07614500 ice cream, icecream
  5: n07584110 consomme
  === images/000000000612.pt ===
  1: n03388549 four-poster
  2: n04429376 throne
  3: n04033995 quilt, comforter, comfort, puff
  4: n02699494 altar
  5: n02100735 English setter
