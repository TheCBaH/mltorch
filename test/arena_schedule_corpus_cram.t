Memory-aware scheduling over every tracked model.json in the pinned producer
submodule, payload-free. Gated on ARENA_EVAL_MODELS; run with
`make arena.schedule.eval`. See bin/arena_schedule_eval.ml.

The original order is always a candidate and wins every tie, so no model's pool
may grow. "constructive" places the original and the two greedy orders;
"beam10k" adds the best beam result at width 8 and 10000 expansions. Pool bytes
are the production planner's witnessed placement of the intermediates.

  $ ../bin/arena_schedule_eval.exe --models-dir "$ARENA_EVAL_MODELS" \
  >   --expected-models 100 \
  >   --settings constructive=1:0,beam10k=8:10000 \
  >   --output eval.jsonl --summary summary.md 2>/dev/null

Every model accounted for, per configuration.

  $ grep '"type":"coverage"' eval.jsonl
  {"type":"coverage","setting":"constructive","expected":100,"attempted":100,"evaluated":100,"refused":0,"failed":0}
  {"type":"coverage","setting":"beam10k","expected":100,"attempted":100,"evaluated":100,"refused":0,"failed":0}

The summary, with every model's pool bytes.

  $ grep -v "^$" summary.md
  # Memory scheduling evaluation
  Pool bytes are the allocated bytes of the production planner's witnessed placement; the original order is always a candidate. No timings, no weights, no RSS claim.
  ## constructive
  - models: 100; pool reduced: 12, unchanged: 88, regressed: 0
  - payload winner would have grown the pool: 3
  - pool bytes, paired total: 1029880528 -> 981965104 (4.65%)
  - target payload peak, paired total: 1000075984 -> 956690448 (4.34%)
  | model | nodes | pool before | pool after | saved | strategy |
  | --- | ---: | ---: | ---: | ---: | --- |
  | mixnet_xxl | 1003 | 44957696 | 36528128 | 18.75% | peak-first |
  | ghostnetv3_130 | 985 | 16457728 | 8429568 | 48.78% | peak-first |
  | mixnet_xl | 976 | 32112640 | 26091520 | 18.75% | peak-first |
  | mixnet_l | 782 | 25690112 | 20873216 | 18.75% | peak-first |
  | tf_mixnet_l | 797 | 25690112 | 20873216 | 18.75% | peak-first |
  | mvitv2_tiny | 798 | 21275920 | 17061136 | 19.81% | peak-first |
  | convit_tiny | 610 | 11446464 | 7721216 | 32.54% | peak-first |
  | ghostnetv3_050 | 985 | 6171648 | 3165184 | 48.71% | live-first |
  | nf_regnet_b0 | 509 | 21508320 | 18707200 | 13.02% | live-first |
  | hiera_tiny_224 | 278 | 16859136 | 15052800 | 10.71% | peak-first |
  | edgenext_xx_small | 277 | 3014224 | 2863696 | 4.99% | peak-first |
  | csatv2 | 979 | 1404928 | 1306624 | 7.00% | peak-first |
  ## beam10k
  - models: 100; pool reduced: 12, unchanged: 88, regressed: 0
  - payload winner would have grown the pool: 3
  - pool bytes, paired total: 1029880528 -> 981965104 (4.65%)
  - target payload peak, paired total: 1000075984 -> 956690448 (4.34%)
  | model | nodes | pool before | pool after | saved | strategy |
  | --- | ---: | ---: | ---: | ---: | --- |
  | mixnet_xxl | 1003 | 44957696 | 36528128 | 18.75% | peak-first |
  | ghostnetv3_130 | 985 | 16457728 | 8429568 | 48.78% | peak-first |
  | mixnet_xl | 976 | 32112640 | 26091520 | 18.75% | peak-first |
  | mixnet_l | 782 | 25690112 | 20873216 | 18.75% | peak-first |
  | tf_mixnet_l | 797 | 25690112 | 20873216 | 18.75% | peak-first |
  | mvitv2_tiny | 798 | 21275920 | 17061136 | 19.81% | peak-first |
  | convit_tiny | 610 | 11446464 | 7721216 | 32.54% | peak-first |
  | ghostnetv3_050 | 985 | 6171648 | 3165184 | 48.71% | live-first |
  | nf_regnet_b0 | 509 | 21508320 | 18707200 | 13.02% | live-first |
  | hiera_tiny_224 | 278 | 16859136 | 15052800 | 10.71% | peak-first |
  | edgenext_xx_small | 277 | 3014224 | 2863696 | 4.99% | peak-first |
  | csatv2 | 979 | 1404928 | 1306624 | 7.00% | peak-first |
  ## All models (pool bytes)
  | model | nodes | original | constructive | beam10k |
  | --- | ---: | ---: | ---: | ---: |
  | bat_resnext26ts | 459 | 12583888 | 12583888 | 12583888 |
  | convit_tiny | 610 | 11446464 | 7721216 | 7721216 |
  | convmixer_1024_20_ks9_p14 | 147 | 3145728 | 3145728 | 3145728 |
  | csatv2 | 979 | 1404928 | 1306624 | 1306624 |
  | darknet17 | 73 | 12845056 | 12845056 | 12845056 |
  | eca_halonext26ts | 281 | 12582912 | 12582912 | 12582912 |
  | edgenext_xx_small | 277 | 3014224 | 2863696 | 2863696 |
  | efficientnet_b0 | 254 | 9633792 | 9633792 | 9633792 |
  | efficientnet_b0_g16_evos | 254 | 9633792 | 9633792 | 9633792 |
  | efficientnet_b0_g8_gn | 303 | 9633792 | 9633792 | 9633792 |
  | efficientnet_b1_pruned | 377 | 7225344 | 7225344 | 7225344 |
  | efficientnet_b2_pruned | 377 | 8132608 | 8132608 | 8132608 |
  | efficientnet_b3_g8_gn | 490 | 19267584 | 19267584 | 19267584 |
  | efficientnet_b3_pruned | 425 | 4816896 | 4816896 | 4816896 |
  | efficientnet_el | 140 | 12845056 | 12845056 | 12845056 |
  | efficientnet_es | 95 | 9633792 | 9633792 | 9633792 |
  | efficientnet_lite2 | 124 | 9633792 | 9633792 | 9633792 |
  | efficientvit_b0 | 245 | 3211264 | 3211264 | 3211264 |
  | fastvit_sa12 | 366 | 9633792 | 9633792 | 9633792 |
  | fastvit_t8 | 305 | 7225344 | 7225344 | 7225344 |
  | fbnetc_100 | 127 | 9633792 | 9633792 | 9633792 |
  | fbnetv3_g | 442 | 9633792 | 9633792 | 9633792 |
  | ghostnetv2_100 | 525 | 5117952 | 5117952 | 5117952 |
  | ghostnetv2_130 | 525 | 6823936 | 6823936 | 6823936 |
  | ghostnetv2_160 | 525 | 8132608 | 8132608 | 8132608 |
  | ghostnetv3_050 | 985 | 6171648 | 3165184 | 3165184 |
  | ghostnetv3_130 | 985 | 16457728 | 8429568 | 8429568 |
  | hgnetv2_b0 | 217 | 3211264 | 3211264 | 3211264 |
  | hiera_tiny_224 | 278 | 16859136 | 15052800 | 15052800 |
  | inception_next_atto | 262 | 4517888 | 4517888 | 4517888 |
  | inception_next_tiny | 382 | 10838016 | 10838016 | 10838016 |
  | inception_v3 | 305 | 6084864 | 6084864 | 6084864 |
  | lambda_resnet26t | 165 | 3145728 | 3145728 | 3145728 |
  | lcnet_035 | 79 | 1605632 | 1605632 | 1605632 |
  | legacy_seresnext26_32x4d | 154 | 9634816 | 9634816 | 9634816 |
  | maxxvitv2_nano_rw_256 | 260 | 14155776 | 14155776 | 14155776 |
  | mixnet_l | 782 | 25690112 | 20873216 | 20873216 |
  | mixnet_xl | 976 | 32112640 | 26091520 | 26091520 |
  | mixnet_xxl | 1003 | 44957696 | 36528128 | 36528128 |
  | mnasnet_140 | 100 | 7225344 | 7225344 | 7225344 |
  | mobilenet_edgetpu_v2_l | 103 | 7225344 | 7225344 | 7225344 |
  | mobilenet_edgetpu_v2_m | 103 | 7225344 | 7225344 | 7225344 |
  | mobilenetv1_100 | 57 | 6422528 | 6422528 | 6422528 |
  | mobilenetv1_125 | 57 | 8028160 | 8028160 | 8028160 |
  | mobilenetv2_050 | 100 | 4816896 | 4816896 | 4816896 |
  | mobilenetv3_large_150d | 238 | 9633792 | 9633792 | 9633792 |
  | mobilenetv3_small_050 | 159 | 1605632 | 1605632 | 1605632 |
  | mobilenetv4_conv_blur_medium | 152 | 19267584 | 19267584 | 19267584 |
  | mobilenetv4_conv_small_050 | 89 | 3211264 | 3211264 | 3211264 |
  | mobilenetv5_base | 1137 | 19280128 | 19280128 | 19280128 |
  | mobileone_s2 | 233 | 9633792 | 9633792 | 9633792 |
  | mobileone_s3 | 233 | 9633792 | 9633792 | 9633792 |
  | mobilevitv2_175 | 278 | 22478848 | 22478848 | 22478848 |
  | mvitv2_tiny | 798 | 21275920 | 17061136 | 17061136 |
  | nf_regnet_b0 | 509 | 21508320 | 18707200 | 18707200 |
  | rdnet_tiny | 319 | 21676032 | 21676032 | 21676032 |
  | regnetx_002 | 101 | 3211264 | 3211264 | 3211264 |
  | regnetx_006 | 122 | 5419008 | 5419008 | 5419008 |
  | regnetx_016 | 136 | 8128512 | 8128512 | 8128512 |
  | regnetx_040 | 171 | 9031680 | 9031680 | 9031680 |
  | regnety_004 | 282 | 5419008 | 5419008 | 5419008 |
  | regnety_008_tv | 248 | 7225344 | 7225344 | 7225344 |
  | regnetz_005 | 340 | 6422528 | 6422528 | 6422528 |
  | regnetz_040 | 451 | 19267584 | 19267584 | 19267584 |
  | regnetz_d8 | 393 | 7226368 | 7226368 | 7226368 |
  | repghostnet_058 | 270 | 2609152 | 2609152 | 2609152 |
  | repghostnet_111 | 270 | 4566016 | 4566016 | 4566016 |
  | repghostnet_130 | 270 | 5218304 | 5218304 | 5218304 |
  | repghostnet_150 | 270 | 5870592 | 5870592 | 5870592 |
  | repvit_m1_0 | 339 | 3512320 | 3512320 | 3512320 |
  | resnest14d | 120 | 9633792 | 9633792 | 9633792 |
  | resnetblur18 | 65 | 6657024 | 6657024 | 6657024 |
  | rexnet_100 | 286 | 9633792 | 9633792 | 9633792 |
  | rexnet_130 | 286 | 12646400 | 12646400 | 12646400 |
  | rexnet_150 | 286 | 14450688 | 14450688 | 14450688 |
  | rexnet_200 | 286 | 19267584 | 19267584 | 19267584 |
  | rexnet_300 | 286 | 28901376 | 28901376 | 28901376 |
  | rexnetr_150 | 286 | 14450688 | 14450688 | 14450688 |
  | rexnetr_200 | 286 | 19267584 | 19267584 | 19267584 |
  | rexnetr_300 | 286 | 28901376 | 28901376 | 28901376 |
  | sam2_hiera_tiny | 299 | 15052800 | 15052800 | 15052800 |
  | sequencer2d_s | 351 | 6340608 | 6340608 | 6340608 |
  | skresnet18 | 229 | 6422528 | 6422528 | 6422528 |
  | starnet_s050 | 58 | 3211264 | 3211264 | 3211264 |
  | starnet_s100 | 74 | 3264512 | 3264512 | 3264512 |
  | swiftformer_xs | 300 | 5419008 | 5419008 | 5419008 |
  | test_convnext2 | 49 | 1843200 | 1843200 | 1843200 |
  | test_efficientnet_gn | 58 | 1228800 | 1228800 | 1228800 |
  | test_vit4 | 191 | 345600 | 345600 | 345600 |
  | tf_efficientnet_es | 108 | 9633792 | 9633792 | 9633792 |
  | tf_efficientnet_lite3 | 155 | 21676032 | 21676032 | 21676032 |
  | tf_mixnet_l | 797 | 25690112 | 20873216 | 20873216 |
  | tf_mixnet_s | 636 | 14450688 | 14450688 | 14450688 |
  | tf_mobilenetv3_large_minimal_100 | 105 | 6541568 | 6541568 | 6541568 |
  | tf_mobilenetv3_small_100 | 171 | 2713600 | 2713600 | 2713600 |
  | tf_mobilenetv3_small_minimal_100 | 81 | 2713600 | 2713600 | 2713600 |
  | vit_small_patch16_dinov3_qkvb | 544 | 6496256 | 6496256 | 6496256 |
  | vit_tiny_r_s16_p8_224 | 240 | 6541568 | 6541568 | 6541568 |
  | volo_d1_224 | 440 | 6422528 | 6422528 | 6422528 |
  | xception41p | 153 | 14450688 | 14450688 | 14450688 |
