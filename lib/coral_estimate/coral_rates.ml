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
    ("class.f32_exp", 453.662);
    ("class.f32_log", 417.334);
    ("class.f32_op", 5.446);
    ("class.f32_sin", 272.558);
    ("class.f32_sqrt", 23.893);
    ("class.int_op", 14.456);
    ("class.load_dtcm", 4.115);
    ("class.load_ram", 6.590);
    ("class.store_dtcm", 3.105);
    ("class.store_ram", 7.285);
    ("class.vec_cos", 353.149);
    ("class.vec_exp", 583.525);
    ("class.vec_f32_div", 9.185);
    ("class.vec_load_dtcm", 3.206);
    ("class.vec_load_ram", 6.438);
    ("class.vec_load_strided_dtcm", 6.480);
    ("class.vec_load_strided_ram", 19.323);
    ("class.vec_log", 427.899);
    ("class.vec_other", 58.430);
    ("class.vec_sin", 347.649);
    ("class.vec_sqrt", 12.114);
    ("class.vec_store_dtcm", 3.568);
    ("class.vec_store_ram", 7.260);
    ("overlap", 0.622);
    ("zvt.bf16.cols_k.ram", 4.140);
    ("zvt.bf16.packed.call", 3215.104);
    ("zvt.bf16.packed.cols_k.ram", 0.384);
    ("zvt.bf16.packed.rows_k.dtcm", 0.322);
    ("zvt.bf16.packed.rows_k.ram", 1.027);
    ("zvt.bf16.packed.tiles.ram", 415.224);
    ("zvt.bf16.rows_k.dtcm", 1.138);
    ("zvt.bf16.rows_k.ram", 4.155);
    ("zvt.bf16.tiles.ram", 269.670);
    ("zvt.f32.cols_k.dtcm", 0.985);
    ("zvt.f32.cols_k.ram", 4.006);
    ("zvt.f32.rows_k.dtcm", 1.249);
    ("zvt.f32.rows_k.ram", 4.270);
    ("zvt.f32.tiles.dtcm", 267.910);
    ("zvt.f32.tiles.ram", 350.773);
  ]
