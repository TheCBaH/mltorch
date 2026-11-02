(* Fitted by test/coral_estimate (NNLS on relative error, the memory classes pulled toward the costs the
   waveforms show) to the VME core's RTL
   calibration data in test/coral_estimate_rtl, v0.0.8 release; the suite
   fails when a fresh fit of that data no longer gives these. Cycles per unit
   of each Coral_model feature; [overlap] is the fraction of the shorter of a
   loop body's vector operations and external vector accesses that the core
   hides under the other. *)

let vme : Coral_model.Rates.t =
  [
    ("attend.dwork.ram", 0.371);
    ("attend.out.ram", 101.997);
    ("attend.qblocks.dtcm", 6372.098);
    ("attend.qblocks.ram", 2699.385);
    ("attend.scores", 57.066);
    ("attend.stage_q.dtcm", 9.054);
    ("attend.stage_q.ram", 4.499);
    ("class.f32_cos", 275.490);
    ("class.f32_div", 16.037);
    ("class.f32_exp", 452.307);
    ("class.f32_log", 415.947);
    ("class.f32_op", 4.395);
    ("class.f32_sin", 271.171);
    ("class.f32_sqrt", 22.512);
    ("class.int_op", 16.356);
    ("class.iteration", 0.947);
    ("class.load_dtcm", 4.104);
    ("class.load_ram", 6.598);
    ("class.local", 23.165);
    ("class.store_dtcm", 3.100);
    ("class.store_ram", 7.290);
    ("class.vec_cos", 353.400);
    ("class.vec_exp", 584.352);
    ("class.vec_f32_div", 7.728);
    ("class.vec_load_dtcm", 3.837);
    ("class.vec_load_ram", 6.780);
    ("class.vec_load_strided_dtcm", 6.331);
    ("class.vec_load_strided_ram", 20.046);
    ("class.vec_log", 428.141);
    ("class.vec_other", 22.223);
    ("class.vec_sin", 347.900);
    ("class.vec_sqrt", 12.169);
    ("class.vec_store_dtcm", 3.971);
    ("class.vec_store_ram", 7.969);
    ("invocation", 236.026);
    ("overlap", 0.721);
    ("zvt.bf16.cols_k.ram", 4.022);
    ("zvt.bf16.packed.call", 3278.758);
    ("zvt.bf16.packed.cols_k.ram", 0.426);
    ("zvt.bf16.packed.rows_k.dtcm", 0.259);
    ("zvt.bf16.packed.rows_k.ram", 0.958);
    ("zvt.bf16.packed.tiles.ram", 471.970);
    ("zvt.bf16.rows_k.dtcm", 1.240);
    ("zvt.bf16.rows_k.ram", 4.239);
    ("zvt.bf16.tiles.ram", 336.745);
    ("zvt.f32.cols_k.dtcm", 0.884);
    ("zvt.f32.cols_k.ram", 3.920);
    ("zvt.f32.rows_k.dtcm", 1.355);
    ("zvt.f32.rows_k.ram", 4.373);
    ("zvt.f32.tiles.dtcm", 305.356);
    ("zvt.f32.tiles.ram", 369.785);
  ]
