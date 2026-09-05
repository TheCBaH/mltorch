(* Public compatibility façade.  The expression representation and each
   operation now live in acyclic, library-private compilation units.  The
   [Expr] surface remains the stable entry point for consumers. *)

module Axis = Expr_internal.Axis
module Role = Expr_internal.Role
module Coord = Expr_internal.Coord
module Source = Expr_internal.Source
module Max_op = Expr_internal.Max_op

type index_op = Expr_internal.Checked.index_op

module Index_overflow = Expr_internal.Checked.Index_overflow
module Reduce_var = Expr_internal.Reduce_var
module Local_var = Expr_internal.Local_var
module Index = Expr_internal.Index
module Intrinsic = Expr_internal.Intrinsic
module Bool = Expr_internal.Bool
module Reduction = Expr_internal.Reduction
module Scan = Expr_internal.Scan
module Value = Expr_internal.Value
module Scan_limits = Expr_internal.Scan_limits
module Scan_meter = Expr_internal.Scan_meter
module Scan_admission = Expr_internal.Scan_admission
module Builder = Expr_internal.Builder
module Fold = Expr_internal.Fold
module Rewrite = Expr_internal.Rewrite
module Check = Expr_internal.Check
module Eval = Expr_internal.Eval
module Pp = Expr_internal.Pp
