import { NextRequest, NextResponse } from "next/server";
import { API_BASE_URL } from "@/lib/config";

export async function GET(req: NextRequest) {
  const token = req.cookies.get("stetho_token")?.value;

  if (!token) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      // Clear token if expired or invalid
      const response = NextResponse.json(
        { error: "Session expired or invalid" },
        { status: 401 }
      );
      response.cookies.set("stetho_token", "", { maxAge: 0, path: "/" });
      return response;
    }

    const user = await res.json();
    return NextResponse.json({ user });
  } catch (error) {
    return NextResponse.json(
      { error: "Authentication service unavailable" },
      { status: 502 }
    );
  }
}

