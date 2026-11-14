(** The C function an emitter produces and the helpers and scratch it needs: the
    one record every C emitter returns. *)

type t = {
  source : string;  (** the function definition *)
  prelude : string;
      (** text the definition needs before it that is not a runtime helper (an
          emitter's own types and functions), made of blocks guarded so that
          several kernels' preludes can sit in one unit: empty for this module's
          own kernels *)
  helpers : Loop_c_runtime.Name.t list;  (** the helpers it calls *)
  local_doubles : int64;
      (** [double]s of scratch [local] must provide: the sum of the program's
          [Alloc] sizes (each gets its own region, so nested or repeated
          allocations can never overlap) *)
  precision : Loop_numerics.Precision.t;
      (** the working precision the kernel was emitted in *)
  refusal : Loop_numerics.Refusal.t option;
      (** why an fp32 policy left it binary64, when admission was the reason *)
  buffer_types : string list;
      (** the C cell type each buffer argument points to, in order *)
}
