(* The peak kernels, from {!Coralnpu_kernels.Peak_kernels}, under the names
   the peak suite and its byte baseline use; [zvt] and [build], the Zvt
   extension and the lowering with it, come from coral-isa's side. *)

include Coralnpu_kernels.Peak_kernels

let zvt = Coralnpu_kernels_zvt.Peak_zvt.zvt
let build = Coralnpu_kernels_zvt.Peak_zvt.build
