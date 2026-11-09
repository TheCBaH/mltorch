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
    ("attend.scores", 56.941);
    ("attend.stage_q.dtcm", 7.934);
    ("attend.stage_q.ram", 3.002);
    ("class.f32_cos", 276.250);
    ("class.f32_div", 16.386);
    ("class.f32_exp", 181.617);
    ("class.f32_log", 416.705);
    ("class.f32_op", 4.740);
    ("class.f32_sin", 271.929);
    ("class.f32_sqrt", 23.265);
    ("class.int_op", 14.201);
    ("class.iteration", 0.586);
    ("class.load_dtcm", 4.125);
    ("class.load_ram", 5.669);
    ("class.store_dtcm", 3.110);
    ("class.store_ram", 7.262);
    ("class.vec_cos", 353.149);
    ("class.vec_exp", 191.738);
    ("class.vec_f32_div", 9.130);
    ("class.vec_load_dtcm", 3.231);
    ("class.vec_load_ram", 6.070);
    ("class.vec_load_strided_dtcm", 6.614);
    ("class.vec_load_strided_ram", 9.470);
    ("class.vec_log", 427.899);
    ("class.vec_other", 13.226);
    ("class.vec_sin", 347.649);
    ("class.vec_sqrt", 11.924);
    ("class.vec_store_dtcm", 3.578);
    ("class.vec_store_ram", 6.999);
    ("class.vec_store_strided_dtcm", 10.337);
    ("overlap", 0.635);
    ("zvt.bf16.cols_k.ram", 1.480);
    ("zvt.bf16.packed.call", 3125.873);
    ("zvt.bf16.packed.cols_k.dtcm", 0.009);
    ("zvt.bf16.packed.cols_k.ram", 0.000);
    ("zvt.bf16.packed.rows_k.dtcm", 0.490);
    ("zvt.bf16.packed.rows_k.ram", 0.553);
    ("zvt.bf16.packed.tiles.ram", 134.873);
    ("zvt.bf16.rows_k.dtcm", 0.939);
    ("zvt.bf16.rows_k.ram", 1.098);
    ("zvt.bf16.tiles.ram", 75.984);
    ("zvt.f32.cols_k.dtcm", 1.473);
    ("zvt.f32.cols_k.ram", 1.582);
    ("zvt.f32.rows_k.dtcm", 0.819);
    ("zvt.f32.rows_k.ram", 0.998);
    ("zvt.f32.tiles.dtcm", 101.451);
    ("zvt.f32.tiles.ram", 165.507);
  ]
