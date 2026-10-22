(** A validated kernel: the one route from a list of items to lowering. *)

module Name_kind : sig
  (** Alphabetical. *)
  type t = Export | Kernel | Label

  val pp : t Fmt.t
end

module Invalid : sig
  (** One thing wrong with a kernel. Alphabetical. *)
  type t =
    | Alignment of { at : Rv32_kernel_item.At.t; boundary : int }
        (** Not a positive power of two, as written. *)
    | Duplicate_export of {
        name : string;
        first : Rv32_kernel_item.At.t;
        second : Rv32_kernel_item.At.t;
      }
    | Duplicate_label of {
        label : Rv32_kernel_label.t;
        first : Rv32_kernel_item.At.t;
        second : Rv32_kernel_item.At.t;
      }
    | Export_twice of {
        label : Rv32_kernel_label.t;
        first : Rv32_kernel_item.At.t;
        second : Rv32_kernel_item.At.t;
      }  (** A label has one export name. *)
    | Export_undefined of {
        name : string;
        label : Rv32_kernel_label.t;
        at : Rv32_kernel_item.At.t;
      }
    | Immediate_range of {
        at : Rv32_kernel_item.At.t;
        range : Rv32_kernel_imm.Imm_range.t;
      }
    | Invalid_name of {
        kind : Name_kind.t;
        name : string;
        at : Rv32_kernel_item.At.t option;  (** [None] for the kernel's name. *)
      }
    | Unresolved_label of {
        label : Rv32_kernel_label.t;
        at : Rv32_kernel_item.At.t;
      }
    | Vsetvli_keeps_vl of { at : Rv32_kernel_item.At.t }
        (** [vsetvli x0, x0, ...] keeps [vl]: not a VLMAX request. *)

  val pp : t Fmt.t
end

type error =
  [ `Invalid of Invalid.t  (** The only failure. *)
  | `Kernel_invalid of Invalid.t list  (** Several, in item order. *) ]

val pp_error : error Fmt.t

type t

val finalize : name:string -> Rv32_kernel_item.t list -> (t, error) Err.t
(** Checks every failure rather than the first: names, label definitions and
    references, exports, alignments, every immediate in its domain, and the
    [vsetvli] form. Names are [[A-Za-z_][A-Za-z0-9_]*]. *)

val items : t -> Rv32_kernel_item.t list
val name : t -> string
