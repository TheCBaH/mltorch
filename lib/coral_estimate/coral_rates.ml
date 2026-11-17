(* Fitted by test/coral_estimate (NNLS on relative error, the memory classes pulled toward the costs the
   waveforms show) to the VME core's RTL
   calibration data in test/coral_estimate_rtl, v0.0.10 release; the suite
   fails when a fresh fit of that data no longer gives these. Cycles per unit
   of each Coral_model feature; [overlap] is the fraction of the shorter of a
   loop body's vector operations and external vector accesses that the core
   hides under the other. *)

let vme : Coral_model.Rates.t =
  [
    ("attend.dwork.ram", 0.389);
    ("attend.kblocks.ram", 290.797);
    ("attend.out.ram", 65.100);
    ("attend.qblocks.dtcm", 11408.419);
    ("attend.qblocks.ram", 4809.113);
    ("attend.scores", 57.510);
    ("class.bf16_round", 17.016);
    ("class.check_index", 21.516);
    ("class.f32_cos", 273.381);
    ("class.f32_div", 13.637);
    ("class.f32_exp", 178.747);
    ("class.f32_log", 413.835);
    ("class.f32_op", 1.997);
    ("class.f32_sin", 269.059);
    ("class.f32_sqrt", 20.396);
    ("class.int_div", 22.244);
    ("class.iteration", 3.333);
    ("class.load_dtcm", 4.120);
    ("class.load_ram", 5.656);
    ("class.store_dtcm", 3.109);
    ("class.store_ram", 7.227);
    ("class.vec_cos", 351.730);
    ("class.vec_exp", 192.589);
    ("class.vec_f32_div", 7.578);
    ("class.vec_load_dtcm", 3.542);
    ("class.vec_load_ram", 6.269);
    ("class.vec_load_strided_dtcm", 6.271);
    ("class.vec_load_strided_ram", 10.103);
    ("class.vec_log", 426.480);
    ("class.vec_other", 13.198);
    ("class.vec_sin", 346.230);
    ("class.vec_sqrt", 10.685);
    ("class.vec_store_dtcm", 3.818);
    ("class.vec_store_ram", 7.454);
    ("class.vec_store_strided_dtcm", 6.998);
    ("overlap", 0.768);
    ("zvt.bf16.cols_k.ram", 1.365);
    ("zvt.bf16.packed.call", 2662.899);
    ("zvt.bf16.packed.cols_k.dtcm", 0.126);
    ("zvt.bf16.packed.rows_k.dtcm", 0.684);
    ("zvt.bf16.packed.rows_k.ram", 0.776);
    ("zvt.bf16.rows_k.dtcm", 1.038);
    ("zvt.bf16.rows_k.ram", 1.184);
    ("zvt.bf16.tiles.ram", 153.267);
    ("zvt.f32.call", 20.103);
    ("zvt.f32.cols_k.dtcm", 1.431);
    ("zvt.f32.cols_k.ram", 1.542);
    ("zvt.f32.rows_k.dtcm", 0.868);
    ("zvt.f32.rows_k.ram", 1.036);
    ("zvt.f32.tiles.dtcm", 125.102);
    ("zvt.f32.tiles.ram", 189.896);
    ("zvt.f32.view.cols_k.dtcm", 3.045);
    ("zvt.f32.view.rows_k.dtcm", 0.201);
    ("zvt.f32.view.tiles.dtcm", 106.645);
  ]
