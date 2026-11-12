(* The random-walk config spaces of the ops in [Conv_convolution], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Convolution = struct
  open Conv_convolution.Convolution

  (* Reuses Conv2d's config space (a non-transposed convolution is shape-
     identical); only the [params] form differs (Hw fields, transposed=false,
     output_padding=0). *)
  module Walk (L : Walk_core.Limits.S) = struct
    module W = Conv_conv2d_walk.Conv2d.Walk (L)
    include W

    let params (c : W.cfg) : params =
      let { Walk_core.Walk.h = sh; w = sw } = c.stride in
      let { Walk_core.Walk.h = ph; w = pw } = c.pad in
      let { Walk_core.Walk.h = dh; w = dw } = c.dilation in
      {
        stride =
          {
            Op_config.Hw.h = Op_config.Pos.of_int sh;
            w = Op_config.Pos.of_int sw;
          };
        padding =
          {
            Op_config.Hw.h = Op_config.Nonneg.of_int ph;
            w = Op_config.Nonneg.of_int pw;
          };
        dilation =
          {
            Op_config.Hw.h = Op_config.Pos.of_int dh;
            w = Op_config.Pos.of_int dw;
          };
        transposed = false;
        output_padding =
          {
            Op_config.Hw.h = Op_config.Nonneg.of_int 0;
            w = Op_config.Nonneg.of_int 0;
          };
        groups = Op_config.Pos.of_int c.groups;
      }
  end
end
