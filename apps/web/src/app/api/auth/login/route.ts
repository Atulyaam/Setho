import { NextRequest, NextResponse } from "next/server";
import { API_BASE_URL } from "@/lib/config";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { username, password } = body;

    if (!username || !password) {
      return NextResponse.json(
        { error: "Username and password are required" },
        { status: 400 }
      );
    }

    const res = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();

    if (!res.ok) {
      return NextResponse.json(
        { error: data.detail || "Authentication failed" },
        { status: res.status }
      );
    }

    const response = NextResponse.json({
      user: data.user,
      expires_in_seconds: data.expires_in_seconds,
    });

    // Set secure HttpOnly cookie for session token
    response.cookies.set("stetho_token", data.access_token, {
      httpOnly: true,
      secure: req.nextUrl.protocol === "https:",
      sameSite: "lax",
      path: "/",
      maxAge: data.expires_in_seconds || 86400,
    });

    return response;
  } catch (error) {
    return NextResponse.json(
      { error: "Unable to connect to authentication service" },
      { status: 502 }
    );
  }
}

