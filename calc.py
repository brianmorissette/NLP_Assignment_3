from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from sympy import sympify, Symbol, Eq, integrate, diff, solveset, S, latex
from sympy.stats import Uniform, Normal, Bernoulli, E, variance

# Initialize MCP server
mcp = FastMCP("calculator")

def _safe_expr(expr: str):
    """
    Parse a math expression safely into a SymPy object.
    Examples:
      "sin(x) + x^2"  -> use ** for power: x**2
      "exp(-x)*cos(x)"
    """
    expr = expr.replace("^", "**")
    return sympify(expr, convert_xor=True)

def _fmt(obj) -> str:
    """Human-friendly string with LaTeX (optional) for clarity."""
    try:
        return f"{str(obj)}\nLaTeX: {latex(obj)}"
    except Exception:
        return str(obj)

@mcp.tool()
async def evaluate(expression: str) -> str:
    """
    Evaluate a numeric/symbolic expression.
    Args:
        expression: e.g., "2+3*4", "sin(pi/3)", "limit(sin(x)/x, x, 0)"
    """
    try:
        val = _safe_expr(expression)
        # Try to evaluate to a number if possible
        evaluated = val.evalf() if val.free_symbols == set() else val
        return f"Result:\n{_fmt(evaluated)}"
    except Exception as e:
        return f"Error: could not evaluate expression. Details: {e}"

@mcp.tool()
async def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """
    Differentiate an expression wrt a variable.
    Args:
        expression: e.g., "sin(x)*exp(x)"
        variable:   e.g., "x"
        order:      e.g., 2 for second derivative
    """
    try:
        # TODO: (optional) validate order >= 1
        if order < 1:
            return "Error: Order must be at least 1"
        # TODO: create a SymPy symbol from `variable`
        x = Symbol(variable)

        # TODO: parse `expression` safely to a SymPy object using _safe_expr
        f = _safe_expr(expression)

        # TODO: compute the derivative of given order
        derivative = diff(f, x, order)

        # format the result using _fmt and return
        return f"diff^{order}/diff{variable}^{order} of {expression}:\n{_fmt(derivative)}"
    except Exception as e:
        return f"Error: differentiation failed. Details: {e}"


@mcp.tool()
async def integrate_expr(expression: str, variable: str = "x",
                         lower: Optional[str] = None,
                         upper: Optional[str] = None) -> str:
    """
    Integrate an expression (indefinite or definite).
    Args:
        expression: e.g., "exp(-x^2)"
        variable:   e.g., "x"
        lower:      e.g., "0" (omit for indefinite)
        upper:      e.g., "1" (omit for indefinite)
    """
    try:
        # TODO (optional): if exactly one of lower/upper is provided, return a helpful error.
        if (lower is not None and upper is None) or (lower is None and upper is not None):
            return "Error: Exactly one of lower/upper cannot be provided"
        # TODO: create a SymPy symbol from `variable`
        x = Symbol(variable)

        # TODO: parse `expression` safely to a SymPy object using _safe_expr
        f = _safe_expr(expression)

        # TODO: compute the integral
        if lower is not None and upper is not None:
            lower_bound = _safe_expr(lower)
            upper_bound = _safe_expr(upper)
            definite_int = integrate(f, (x, lower_bound, upper_bound))
            return f"∫_{lower}^{upper} {expression} d{variable} =\n{_fmt(definite_int)}"
        else:
            indefinite_int = integrate(f, x)
            return f"∫ {expression} d{variable} =\n{_fmt(indefinite_int)} + C"
    except Exception as e:
        return f"Error: integration failed. Details: {e}"

@mcp.tool()
async def solve_equation(equation: str, variable: str = "x", domain: str = "C") -> str:
    """
    Solve equation(s) for a variable.
    Args:
        equation: e.g., "x^2 - 2 = 0" or "sin(x)=1/2"
        variable: variable to solve for
        domain:   'R' (reals) or 'C' (complex)
    """
    try:
        # TODO: create a SymPy symbol for the variable
        x = Symbol(variable)

        # TODO: handle input string
        # If equation contains "=", split into lhs and rhs
        if "=" in equation:
            lhs, rhs = equation.split("=")
            lhs = _safe_expr(lhs)
            rhs = _safe_expr(rhs)
            expr = Eq(lhs, rhs)
        # Otherwise, treat as "expression = 0" using Eq(...)
        else:
            equation = _safe_expr(equation)
            expr = Eq(equation, S.Zero)

#        TODO: select domain (S.Reals if domain == "R", else S.Complexes)
        dom = S.Reals if domain == "R" else S.Complexes

        # TODO: solve using solveset
        sol = solveset(expr, x, domain=dom)

        return f"Solutions for {equation} over {domain}:\n{_fmt(sol)}"
    except Exception as e:
        return f"Error: solving failed. Details: {e}"

@mcp.tool()
async def stats_distribution(distribution: str, params: str, compute: str = "both") -> str:
    """
    Compute expectation and/or variance for simple probability distributions.
    Args:
        distribution: "Uniform", "Normal", or "Bernoulli"
        params: distribution parameters as comma-separated values:
                - Uniform: "a,b" (lower and upper bounds)
                - Normal: "mu,sigma" (mean and standard deviation)
                - Bernoulli: "p" (success probability)
        compute: "expectation", "variance", or "both" (default)
    """
    try:
        # Parse parameters
        param_list = [p.strip() for p in params.split(",")]
        
        # Create the distribution
        X = None
        if distribution.lower() == "uniform":
            if len(param_list) != 2:
                return "Error: Uniform distribution requires 2 parameters: a,b (lower and upper bounds)"
            a = _safe_expr(param_list[0])
            b = _safe_expr(param_list[1])
            X = Uniform("X", a, b)
            dist_desc = f"Uniform({a}, {b})"
        elif distribution.lower() == "normal":
            if len(param_list) != 2:
                return "Error: Normal distribution requires 2 parameters: mu,sigma (mean and std dev)"
            mu = _safe_expr(param_list[0])
            sigma = _safe_expr(param_list[1])
            X = Normal("X", mu, sigma)
            dist_desc = f"Normal({mu}, {sigma})"
        elif distribution.lower() == "bernoulli":
            if len(param_list) != 1:
                return "Error: Bernoulli distribution requires 1 parameter: p (success probability)"
            p = _safe_expr(param_list[0])
            X = Bernoulli("X", p)
            dist_desc = f"Bernoulli({p})"
        else:
            return f"Error: Unsupported distribution '{distribution}'. Choose from: Uniform, Normal, Bernoulli"
        
        # Compute requested statistics
        result = f"Distribution: {dist_desc}\n"
        
        if compute.lower() in ["expectation", "both"]:
            exp = E(X)
            result += f"\nExpectation E[X]:\n{_fmt(exp)}"
        
        if compute.lower() in ["variance", "both"]:
            var = variance(X)
            result += f"\nVariance Var(X):\n{_fmt(var)}"
        
        if compute.lower() not in ["expectation", "variance", "both"]:
            return f"Error: compute must be 'expectation', 'variance', or 'both'. Got: '{compute}'"
        
        return result
    except Exception as e:
        return f"Error: statistics computation failed. Details: {e}"

def main():
    # Use stdio transport for Claude Desktop integration
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
