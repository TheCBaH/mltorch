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
    ("class.f32_cos", 276.199);
    ("class.f32_div", 16.336);
    ("class.f32_exp", 181.566);
    ("class.f32_log", 416.654);
    ("class.f32_op", 4.690);
    ("class.f32_sin", 271.878);
    ("class.f32_sqrt", 23.213);
    ("class.int_op", 14.194);
    ("class.iteration", 0.635);
    ("class.load_dtcm", 4.125);
    ("class.load_ram", 5.669);
    ("class.store_dtcm", 3.110);
    ("class.store_ram", 7.262);
    ("class.vec_cos", 352.957);
    ("class.vec_exp", 191.740);
    ("class.vec_f32_div", 9.120);
    ("class.vec_load_dtcm", 3.230);
    ("class.vec_load_ram", 6.070);
    ("class.vec_load_strided_dtcm", 6.601);
    ("class.vec_load_strided_ram", 9.460);
    ("class.vec_log", 427.707);
    ("class.vec_other", 11.964);
    ("class.vec_sin", 347.457);
    ("class.vec_sqrt", 11.913);
    ("class.vec_store_dtcm", 3.577);
    ("class.vec_store_ram", 6.998);
    ("class.vec_store_strided_dtcm", 11.501);
    ("overlap", 0.636);
    ("zvt.bf16.cols_k.ram", 1.480);
    ("zvt.bf16.packed.call", 3131.113);
    ("zvt.bf16.packed.cols_k.dtcm", 0.009);
    ("zvt.bf16.packed.cols_k.ram", 0.000);
    ("zvt.bf16.packed.rows_k.dtcm", 0.490);
    ("zvt.bf16.packed.rows_k.ram", 0.553);
    ("zvt.bf16.packed.tiles.ram", 134.866);
    ("zvt.bf16.rows_k.dtcm", 0.940);
    ("zvt.bf16.rows_k.ram", 1.099);
    ("zvt.bf16.tiles.ram", 75.907);
    ("zvt.f32.cols_k.dtcm", 1.478);
    ("zvt.f32.cols_k.ram", 1.587);
    ("zvt.f32.rows_k.dtcm", 0.815);
    ("zvt.f32.rows_k.ram", 0.994);
    ("zvt.f32.tiles.dtcm", 100.236);
    ("zvt.f32.tiles.ram", 164.879);
  ]
