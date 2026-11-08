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
    ("class.f32_cos", 275.750);
    ("class.f32_div", 15.986);
    ("class.f32_exp", 181.117);
    ("class.f32_log", 416.205);
    ("class.f32_op", 4.339);
    ("class.f32_sin", 271.429);
    ("class.f32_sqrt", 22.764);
    ("class.int_op", 14.288);
    ("class.iteration", 1.076);
    ("class.load_dtcm", 4.115);
    ("class.load_ram", 6.590);
    ("class.store_dtcm", 3.105);
    ("class.store_ram", 7.285);
    ("class.vec_cos", 353.149);
    ("class.vec_exp", 191.978);
    ("class.vec_f32_div", 8.804);
    ("class.vec_load_dtcm", 3.214);
    ("class.vec_load_ram", 6.421);
    ("class.vec_load_strided_dtcm", 6.436);
    ("class.vec_load_strided_ram", 19.128);
    ("class.vec_log", 427.899);
    ("class.vec_other", 13.111);
    ("class.vec_sin", 347.649);
    ("class.vec_sqrt", 11.837);
    ("class.vec_store_dtcm", 3.568);
    ("class.vec_store_ram", 7.207);
    ("class.vec_store_strided_dtcm", 14.120);
    ("overlap", 0.623);
    ("zvt.bf16.cols_k.ram", 4.121);
    ("zvt.bf16.packed.call", 3390.504);
    ("zvt.bf16.packed.cols_k.ram", 0.387);
    ("zvt.bf16.packed.rows_k.dtcm", 0.312);
    ("zvt.bf16.packed.rows_k.ram", 1.017);
    ("zvt.bf16.packed.tiles.ram", 415.224);
    ("zvt.bf16.rows_k.dtcm", 1.160);
    ("zvt.bf16.rows_k.ram", 4.180);
    ("zvt.bf16.tiles.ram", 271.369);
    ("zvt.f32.cols_k.dtcm", 1.179);
    ("zvt.f32.cols_k.ram", 4.168);
    ("zvt.f32.rows_k.dtcm", 1.109);
    ("zvt.f32.rows_k.ram", 4.157);
    ("zvt.f32.tiles.dtcm", 231.173);
    ("zvt.f32.tiles.ram", 333.709);
  ]
