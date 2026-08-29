(* The structured statements, parametrized by the region type they nest so the
   region record (which holds statements) can live in its own module named [t].

   [For] and [Ordered_sum] bounds are evaluated once; the interval is
   half-open and the induction value is the first parameter of the body.
   Carried values (the effect among them) are the body's remaining parameters,
   initialised by [inits], and each iteration transfers all yields at once. A
   zero-trip loop returns its initializers without running the body. *)
type 'region t =
  | For of {
      lo : Ssa_value.t;
      hi : Ssa_value.t;
      step : int64;
      inits : Ssa_value.t list;
      results : Ssa_value.t list;
      body : 'region;
    }
  | If of {
      cond : Ssa_value.t;
      results : Ssa_value.t list;
      then_ : 'region;
      else_ : 'region;
    }
  | Instr of Ssa_instr.t
  | Ordered_sum of {
      lo : Ssa_value.t;
      hi : Ssa_value.t;
      seed : Ssa_value.t;
      token : Ssa_value.t;
      results : Ssa_value.t list;
      body : 'region;
    }
      (** The left fold [acc <- seed; for var in [lo, hi): acc <- acc + term],
          its body yielding [term; effect]; results are [sum; effect]. An empty
          range returns the seed with no term evaluated. *)
