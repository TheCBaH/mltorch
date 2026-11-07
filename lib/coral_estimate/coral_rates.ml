(* Fitted by test/coral_estimate (NNLS on relative error, the memory classes pulled toward the costs the
   waveforms show) to the VME core's RTL
   calibration data in test/coral_estimate_rtl, v0.0.9 release; the suite
   fails when a fresh fit of that data no longer gives these. Cycles per unit
   of each Coral_model feature; [overlap] is the fraction of the shorter of a
   loop body's vector operations and external vector accesses that the core
   hides under the other. *)

let vme : Coral_model.Rates.t =
  [
    ("attend.dwork.ram", 0.372);
    ("attend.out.ram", 98.456);
    ("attend.qblocks.dtcm", 6570.839);
    ("attend.qblocks.ram", 3082.995);
    ("attend.scores", 56.941);
    ("attend.stage_q.dtcm", 9.162);
    ("attend.stage_q.ram", 4.418);
    ("class.f32_cos", 276.879);
    ("class.f32_div", 17.081);
    ("class.f32_exp", 451.507);
    ("class.f32_log", 417.333);
    ("class.f32_op", 5.444);
    ("class.f32_sin", 272.557);
    ("class.f32_sqrt", 23.893);
    ("class.int_op", 14.456);
    ("class.iteration", 0.948);
    ("class.load_dtcm", 4.115);
    ("class.load_ram", 6.591);
    ("class.local", 0.000);
    ("class.store_dtcm", 3.100);
    ("class.store_ram", 7.286);
    ("class.vec_cos", 353.147);
    ("class.vec_exp", 582.690);
    ("class.vec_f32_div", 9.180);
    ("class.vec_load_dtcm", 3.207);
    ("class.vec_load_ram", 6.438);
    ("class.vec_load_strided_dtcm", 6.487);
    ("class.vec_load_strided_ram", 19.324);
    ("class.vec_log", 427.897);
    ("class.vec_other", 57.556);
    ("class.vec_sin", 347.647);
    ("class.vec_sqrt", 12.112);
    ("class.vec_store_dtcm", 3.568);
    ("class.vec_store_ram", 7.260);
    ("invocation", 0.000);
    ("overlap", 0.621);
    ("zvt.bf16.cols_k.ram", 4.140);
    ("zvt.bf16.packed.call", 3218.476);
    ("zvt.bf16.packed.cols_k.ram", 0.384);
    ("zvt.bf16.packed.rows_k.dtcm", 0.322);
    ("zvt.bf16.packed.rows_k.ram", 1.027);
    ("zvt.bf16.packed.tiles.ram", 415.253);
    ("zvt.bf16.rows_k.dtcm", 1.139);
    ("zvt.bf16.rows_k.ram", 4.156);
    ("zvt.bf16.tiles.ram", 269.836);
    ("zvt.f32.cols_k.dtcm", 0.989);
    ("zvt.f32.cols_k.ram", 4.009);
    ("zvt.f32.rows_k.dtcm", 1.247);
    ("zvt.f32.rows_k.ram", 4.268);
    ("zvt.f32.tiles.dtcm", 267.387);
    ("zvt.f32.tiles.ram", 378.847);
  ]
