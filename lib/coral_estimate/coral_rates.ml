(* Fitted by test/coral_estimate (NNLS on relative error, the memory classes pulled toward the costs the
   waveforms show) to the VME core's RTL
   calibration data in test/coral_estimate_rtl, v0.0.6 release; the suite
   fails when a fresh fit of that data no longer gives these. Cycles per unit
   of each Coral_model feature. *)

let vme : Coral_model.Rates.t =
  [
    ("attend.dwork.ram", 0.37);
    ("attend.out.ram", 102.516);
    ("attend.qblocks.dtcm", 6342.991);
    ("attend.qblocks.ram", 2643.202);
    ("attend.scores", 57.084);
    ("attend.stage_q.dtcm", 9.038);
    ("attend.stage_q.ram", 4.511);
    ("class.f32_cos", 275.672);
    ("class.f32_div", 16.27);
    ("class.f32_exp", 452.489);
    ("class.f32_log", 416.129);
    ("class.f32_op", 4.628);
    ("class.f32_sin", 271.353);
    ("class.f32_sqrt", 22.694);
    ("class.int_op", 16.38);
    ("class.iteration", 0.694);
    ("class.load_dtcm", 4.104);
    ("class.load_ram", 6.599);
    ("class.local", 22.969);
    ("class.store_dtcm", 3.099);
    ("class.store_ram", 7.294);
    ("class.vec_cos", 353.157);
    ("class.vec_exp", 581.657);
    ("class.vec_f32_div", 4.996);
    ("class.vec_load_dtcm", 3.847);
    ("class.vec_load_ram", 6.735);
    ("class.vec_load_strided_dtcm", 6.357);
    ("class.vec_load_strided_ram", 20.042);
    ("class.vec_log", 427.897);
    ("class.vec_other", 17.164);
    ("class.vec_sin", 347.657);
    ("class.vec_sqrt", 11.952);
    ("class.vec_store_dtcm", 3.979);
    ("class.vec_store_ram", 7.933);
    ("invocation", 270.594);
    ("zvt.bf16.cols_k.ram", 3.963);
    ("zvt.bf16.packed.call", 3268.907);
    ("zvt.bf16.packed.cols_k.ram", 0.426);
    ("zvt.bf16.packed.rows_k.dtcm", 0.26);
    ("zvt.bf16.packed.rows_k.ram", 0.959);
    ("zvt.bf16.packed.tiles.ram", 481.179);
    ("zvt.bf16.rows_k.dtcm", 1.302);
    ("zvt.bf16.rows_k.ram", 4.303);
    ("zvt.bf16.tiles.ram", 339.54);
    ("zvt.f32.cols_k.dtcm", 0.86);
    ("zvt.f32.cols_k.ram", 3.898);
    ("zvt.f32.rows_k.dtcm", 1.375);
    ("zvt.f32.rows_k.ram", 4.392);
    ("zvt.f32.tiles.dtcm", 313.93);
    ("zvt.f32.tiles.ram", 380.462);
  ]
