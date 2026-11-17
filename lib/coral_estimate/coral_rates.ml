(* Fitted by test/coral_estimate (NNLS on relative error, the memory classes pulled toward the costs the
   waveforms show) to the VME core's RTL
   calibration data in test/coral_estimate_rtl, v0.0.10 release; the suite
   fails when a fresh fit of that data no longer gives these. Cycles per unit
   of each Coral_model feature; [overlap] is the fraction of the shorter of a
   loop body's vector operations and external vector accesses that the core
   hides under the other. *)

let vme : Coral_model.Rates.t =
  [
    ("attend.dwork.ram", 0.338);
    ("attend.out.ram", 98.356);
    ("attend.qblocks.dtcm", 7188.629);
    ("attend.qblocks.ram", 3590.839);
    ("attend.scores", 56.956);
    ("attend.stage_q.dtcm", 7.934);
    ("attend.stage_q.ram", 3.002);
    ("class.f32_cos", 276.220);
    ("class.f32_div", 16.357);
    ("class.f32_exp", 181.587);
    ("class.f32_log", 416.675);
    ("class.f32_op", 4.711);
    ("class.f32_sin", 271.899);
    ("class.f32_sqrt", 23.235);
    ("class.int_op", 14.233);
    ("class.iteration", 0.615);
    ("class.load_dtcm", 4.125);
    ("class.load_ram", 5.669);
    ("class.store_dtcm", 3.110);
    ("class.store_ram", 7.262);
    ("class.vec_cos", 352.962);
    ("class.vec_exp", 191.735);
    ("class.vec_f32_div", 9.126);
    ("class.vec_load_dtcm", 3.230);
    ("class.vec_load_ram", 6.069);
    ("class.vec_load_strided_dtcm", 6.602);
    ("class.vec_load_strided_ram", 9.461);
    ("class.vec_log", 427.713);
    ("class.vec_other", 11.957);
    ("class.vec_sin", 347.462);
    ("class.vec_sqrt", 11.918);
    ("class.vec_store_dtcm", 3.577);
    ("class.vec_store_ram", 6.997);
    ("class.vec_store_strided_dtcm", 11.787);
    ("overlap", 0.635);
    ("zvt.bf16.cols_k.ram", 1.480);
    ("zvt.bf16.packed.call", 3130.602);
    ("zvt.bf16.packed.cols_k.dtcm", 0.009);
    ("zvt.bf16.packed.rows_k.dtcm", 0.490);
    ("zvt.bf16.packed.rows_k.ram", 0.553);
    ("zvt.bf16.packed.tiles.ram", 134.701);
    ("zvt.bf16.rows_k.dtcm", 0.939);
    ("zvt.bf16.rows_k.ram", 1.099);
    ("zvt.bf16.tiles.ram", 75.792);
    ("zvt.f32.cols_k.dtcm", 1.478);
    ("zvt.f32.cols_k.ram", 1.587);
    ("zvt.f32.rows_k.dtcm", 0.815);
    ("zvt.f32.rows_k.ram", 0.994);
    ("zvt.f32.tiles.dtcm", 100.273);
    ("zvt.f32.tiles.ram", 164.889);
    ("zvt.f32.view.cols_k.dtcm", 3.504);
    ("zvt.f32.view.tiles.dtcm", 118.384);
  ]
