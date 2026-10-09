(** Run every published case of an opened fixture and report, per output, how it
    compares with the reference.

    The route is Native direct execution ([Native_interp.run_named]); the report
    says so, and a transformed or Kernel result is never implied by it. A case
    whose own digests do not match its descriptor, whose inputs do not match the
    contract, or whose graph cannot run is recorded as such -- never as a pass.
*)

val backend : string
(** ["native-direct"]. *)

val to_logical :
  Tensor.packed ->
  rank:int ->
  (Pt2_fixture.Logical.t, [> `Native_layout of string ]) result
(** A Native tensor in logical coordinates: its dtype, its last [rank] frame
    axes (the leading ones must have extent 1), and its elements in row-major
    order. *)

val replay :
  consumer:string ->
  Pt2_fixture_unix.Fixture.t ->
  (Pt2_fixture.Report.t, [> Pt2_fixture_unix.Fixture.error ]) Err.t
(** [consumer] names the consumer revision and workspace state for the report.
    Errors are those of reading the bundle; everything about the cases is in the
    report. *)
